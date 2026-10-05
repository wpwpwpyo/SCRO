#!/usr/bin/env python3
"""Visualize exact per-vector displacement without nonlinear embedding.

For a target residual ``r_i`` and realized fitted-key update ``delta_k_i``,
define

    d_i = delta_k_i - r_i
    a_i = <d_i, r_i> / ||r_i||^2
    b_i = ||d_i - a_i r_i|| / ||r_i||

Then ``sqrt(a_i**2 + b_i**2)`` is exactly
``||delta_k_i-r_i|| / ||r_i||`` (up to floating-point roundoff). Consequently,
the distance of every plotted point from the origin is an interpretable,
projection-free relative error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager

_TIMES_FONT_DIRECTORY = Path("/usr/share/fonts/truetype/msttcorefonts")
if _TIMES_FONT_DIRECTORY.is_dir():
    for _font_path in _TIMES_FONT_DIRECTORY.glob("Times_New_Roman*.ttf"):
        font_manager.fontManager.addfont(str(_font_path))

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


@dataclass
class DisplacementData:
    name: str
    deltak_path: Path
    r_path: Path
    reference: np.ndarray
    fitted: np.ndarray
    parallel: np.ndarray
    orthogonal: np.ndarray
    relative_error: np.ndarray
    valid_indices: np.ndarray
    invalid_zero_reference_count: int
    identity_max_abs_error: float
    metrics: Dict[str, float]


def _atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _load_vectors(
    path: Path,
    key: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(
                f"{path} is missing {key!r}; available keys={archive.files}"
            )
        case_ids = (
            np.asarray(archive["case_ids"]).reshape(-1)
            if "case_ids" in archive.files
            else None
        )
        layer = (
            int(np.asarray(archive["layer"]).reshape(-1)[0])
            if "layer" in archive.files
            else None
        )
        vectors = np.ascontiguousarray(archive[key], dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"{path}:{key} must be rank 2, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        count = int(vectors.size - np.isfinite(vectors).sum())
        raise ValueError(f"{path}:{key} contains {count} non-finite values")
    if case_ids is not None and case_ids.size != vectors.shape[0]:
        raise ValueError(
            f"{path}: case_ids has {case_ids.size} entries but {key} has "
            f"{vectors.shape[0]} sample rows"
        )
    return vectors, case_ids, layer


def _load_pair(
    deltak_path: Path,
    r_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    fitted, deltak_case_ids, deltak_layer = _load_vectors(
        deltak_path, "deltak"
    )
    reference, r_case_ids, r_layer = _load_vectors(r_path, "r")
    if reference.shape != fitted.shape:
        raise ValueError(
            "Vector shapes differ between the two files: "
            f"reference={reference.shape}, fitted={fitted.shape}"
        )
    if reference.shape[0] == 0 or reference.shape[1] == 0:
        raise ValueError(f"Vector matrices are empty: {reference.shape}")
    if (deltak_case_ids is None) != (r_case_ids is None):
        raise ValueError(
            "case_ids must be present in both files or absent from both files"
        )
    if deltak_case_ids is not None and not np.array_equal(
        deltak_case_ids, r_case_ids
    ):
        raise ValueError("deltak.npz and r.npz have different case_ids/order")
    if (
        deltak_layer is not None
        and r_layer is not None
        and deltak_layer != r_layer
    ):
        raise ValueError(
            f"Layer mismatch: deltak layer={deltak_layer}, r layer={r_layer}"
        )
    return reference, fitted


def _quantile_metrics(prefix: str, values: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(values.max()),
    }


def compute_displacements(
    *,
    name: str,
    deltak_path: Path,
    r_path: Path,
    reference: np.ndarray,
    fitted: np.ndarray,
    zero_norm_epsilon: float,
) -> DisplacementData:
    reference_norm_sq = np.einsum("ij,ij->i", reference, reference)
    valid = reference_norm_sq > zero_norm_epsilon**2
    invalid_count = int((~valid).sum())
    if not valid.any():
        raise ValueError(f"Every reference vector has zero norm in {r_path}")

    reference = reference[valid]
    fitted = fitted[valid]
    reference_norm_sq = reference_norm_sq[valid]
    reference_norm = np.sqrt(reference_norm_sq)
    displacement = fitted - reference
    parallel = np.einsum("ij,ij->i", displacement, reference) / reference_norm_sq
    orthogonal_vectors = displacement - parallel[:, None] * reference
    orthogonal = np.linalg.norm(orthogonal_vectors, axis=1) / reference_norm
    relative_error = np.linalg.norm(displacement, axis=1) / reference_norm
    reconstructed_error = np.sqrt(parallel**2 + orthogonal**2)
    identity_error = float(
        np.max(np.abs(relative_error - reconstructed_error), initial=0.0)
    )

    error_fro_sq = float(np.square(displacement).sum(dtype=np.float64))
    reference_fro_sq = float(np.square(reference).sum(dtype=np.float64))
    batch_relative_error = math.sqrt(error_fro_sq / reference_fro_sq)
    metrics = {
        "num_vectors_total": int(valid.size),
        "num_vectors_valid": int(valid.sum()),
        "num_zero_norm_reference_vectors_excluded": invalid_count,
        "dimension": int(reference.shape[1]),
        "batch_relative_frobenius_error": float(batch_relative_error),
        "error_fro_sq": error_fro_sq,
        "reference_fro_sq": reference_fro_sq,
        "parallel_mean": float(parallel.mean()),
        "parallel_mean_abs": float(np.abs(parallel).mean()),
        "parallel_negative_fraction": float((parallel < 0).mean()),
        **_quantile_metrics("orthogonal_relative", orthogonal),
        **_quantile_metrics("per_vector_relative_error", relative_error),
        "decomposition_identity_max_abs_error": identity_error,
    }
    return DisplacementData(
        name=name,
        deltak_path=deltak_path,
        r_path=r_path,
        reference=reference,
        fitted=fitted,
        parallel=parallel,
        orthogonal=orthogonal,
        relative_error=relative_error,
        valid_indices=np.flatnonzero(valid),
        invalid_zero_reference_count=invalid_count,
        identity_max_abs_error=identity_error,
        metrics=metrics,
    )


def _subsample_indices(
    count: int, maximum: int, seed: int
) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=maximum, replace=False))


def _resolve_axis_limit(
    datasets: Sequence[DisplacementData],
    levels: Sequence[float],
    explicit: float,
    quantile: float,
) -> float:
    if explicit > 0:
        return explicit
    values = np.concatenate([item.relative_error for item in datasets])
    radius = float(np.quantile(values, quantile))
    if levels:
        radius = max(radius, max(levels))
    return max(radius * 1.08, np.finfo(np.float64).eps)


def draw(
    datasets: Sequence[DisplacementData],
    *,
    output: Path,
    levels: Sequence[float],
    axis_max: float,
    axis_quantile: float,
    max_scatter_points: int,
    random_state: int,
    point_size: float,
    point_alpha: float,
    dpi: int,
    layout: Tuple[int, int],
    title_font_size: float,
    axis_font_size: float,
    legend_font_size: float,
    other_font_size: float,
) -> Tuple[Path, Path]:
    radius = _resolve_axis_limit(datasets, levels, axis_max, axis_quantile)
    all_errors = np.concatenate([item.relative_error for item in datasets])
    color_max = max(float(np.quantile(all_errors, 0.99)), radius * 0.25)
    color_norm = Normalize(vmin=0.0, vmax=color_max, clip=True)
    cmap = plt.get_cmap("viridis")

    rows, columns = layout
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.35 * columns, 2.45 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    occupied_axes = flat_axes[: len(datasets)]
    theta = np.linspace(0.0, math.pi, 360)
    for dataset_index, (item, scatter_ax) in enumerate(
        zip(datasets, occupied_axes)
    ):
        indices = _subsample_indices(
            item.relative_error.size,
            max_scatter_points,
            random_state + dataset_index,
        )
        scatter_ax.scatter(
            item.parallel[indices],
            item.orthogonal[indices],
            c=item.relative_error[indices],
            cmap=cmap,
            norm=color_norm,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
            rasterized=True,
        )
        for level in levels:
            scatter_ax.plot(
                level * np.cos(theta),
                level * np.sin(theta),
                color="#6E6E6E",
                linewidth=0.8,
                linestyle=(0, (4, 3)),
                alpha=0.8,
            )
            scatter_ax.text(
                level / math.sqrt(2.0),
                level / math.sqrt(2.0),
                f"{level:g}",
                fontsize=other_font_size,
                color="#555555",
            )
        scatter_ax.axvline(0.0, color="#444444", linewidth=0.8)
        scatter_ax.set_xlim(-radius, radius)
        scatter_ax.set_ylim(0.0, radius)
        scatter_ax.set_aspect("equal", adjustable="box")
        scatter_ax.set_xlabel(
            r"Parallel relative displacement $a_i$",
            fontsize=axis_font_size,
            fontweight="bold",
        )
        scatter_ax.set_ylabel(
            r"Orthogonal relative displacement $b_i$",
            fontsize=axis_font_size,
            fontweight="bold",
        )
        scatter_ax.tick_params(axis="both", labelsize=axis_font_size)
        for tick_label in (
            list(scatter_ax.get_xticklabels())
            + list(scatter_ax.get_yticklabels())
        ):
            tick_label.set_fontweight("bold")
        scatter_ax.grid(True, color="#D8D8D8", linewidth=0.5, alpha=0.65)
        scatter_ax.set_axisbelow(True)
        scatter_ax.set_title(
            item.name,
            fontsize=title_font_size,
            fontweight="bold",
        )
        mean_relative_error = item.metrics["per_vector_relative_error_mean"]
        scatter_ax.text(
            0.97,
            0.95,
            (
                r"$\mathrm{mean}\!\left("
                r"\frac{\|\Delta k_i-r_i\|_2}{\|r_i\|_2}"
                r"\right)$"
                f"\n$={mean_relative_error:.4f}$"
            ),
            transform=scatter_ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=other_font_size,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.5,
            },
        )

    for unused_ax in flat_axes[len(datasets) :]:
        unused_ax.set_axis_off()

    mappable = ScalarMappable(norm=color_norm, cmap=cmap)
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable,
        ax=occupied_axes,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(
        "True per-vector relative error",
        fontsize=legend_font_size,
        fontweight="bold",
    )
    colorbar.ax.tick_params(labelsize=legend_font_size)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    png_output = output.with_suffix(".png")
    pdf_output = output.with_suffix(".pdf")
    png_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved displacement PNG: {png_output}")
    print(f"Saved displacement PDF: {pdf_output}")
    return png_output, pdf_output


def _parse_layout(value: str) -> Tuple[int, int]:
    normalized = value.strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1]
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "layout must be ROWS,COLUMNS, for example '(2,3)'"
        )
    try:
        rows, columns = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layout rows and columns must be integers"
        ) from error
    if rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError(
            "layout rows and columns must both be positive"
        )
    return rows, columns


def _configure_font_sizes(
    *,
    title_font_size: float,
    axis_font_size: float,
    legend_font_size: float,
    other_font_size: float,
) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": other_font_size,
            "axes.titlesize": title_font_size,
            "axes.titleweight": "bold",
            "axes.labelsize": axis_font_size,
            "axes.labelweight": "bold",
            "xtick.labelsize": axis_font_size,
            "ytick.labelsize": axis_font_size,
            "legend.fontsize": legend_font_size,
            "figure.titlesize": title_font_size,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot exact parallel/orthogonal vector displacement from "
            "separate deltak.npz and r.npz diagnostic archives."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("NAME", "DELTAK_NPZ", "R_NPZ"),
        required=True,
        help=(
            "Named pair of NPZ inputs; repeat to compare methods. The files "
            "must contain arrays named 'deltak' and 'r', respectively."
        ),
    )
    parser.add_argument(
        "--layout",
        type=_parse_layout,
        metavar="ROWS,COLUMNS",
        help=(
            "Subplot grid such as '(2,3)' or '2,3'. Inputs fill the grid "
            "in row-major order; unused panels remain blank. The default is "
            "one column with one row per input."
        ),
    )
    parser.add_argument("--zero-norm-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--levels", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5]
    )
    parser.add_argument(
        "--axis-max",
        type=float,
        default=0.0,
        help="Positive fixed radius; 0 chooses it from the data",
    )
    parser.add_argument(
        "--axis-quantile",
        type=float,
        default=1.0,
        help="Automatic radius quantile in (0,1]; default keeps all points",
    )
    parser.add_argument("--max-scatter-points", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--point-alpha", type=float, default=0.45)
    parser.add_argument("--title-font-size", type=float, default=10.0)
    parser.add_argument("--axis-font-size", type=float, default=8.0)
    parser.add_argument("--legend-font-size", type=float, default=8.0)
    parser.add_argument("--other-font-size", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output path or stem. Both same-stem .png and .pdf files are "
            "always generated."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Defaults to <output-stem>.summary.json",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    names = [name for name, _, _ in args.input]
    if len(names) != len(set(names)):
        raise ValueError(f"Input names must be unique: {names}")
    if args.zero_norm_epsilon < 0:
        raise ValueError("zero-norm-epsilon must be non-negative")
    if any(level <= 0 for level in args.levels):
        raise ValueError("every level must be positive")
    if args.axis_max < 0:
        raise ValueError("axis-max must be non-negative")
    if not 0 < args.axis_quantile <= 1:
        raise ValueError("axis-quantile must lie in (0,1]")
    if args.max_scatter_points < 0:
        raise ValueError("max-scatter-points must be non-negative")
    if args.point_size <= 0 or not 0 < args.point_alpha <= 1:
        raise ValueError("invalid point-size or point-alpha")
    font_sizes = {
        "title-font-size": args.title_font_size,
        "axis-font-size": args.axis_font_size,
        "legend-font-size": args.legend_font_size,
        "other-font-size": args.other_font_size,
    }
    for name, value in font_sizes.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    if args.layout is not None:
        rows, columns = args.layout
        if rows * columns < len(args.input):
            raise ValueError(
                f"layout {args.layout} has {rows * columns} panels but "
                f"{len(args.input)} inputs were supplied"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        _configure_font_sizes(
            title_font_size=args.title_font_size,
            axis_font_size=args.axis_font_size,
            legend_font_size=args.legend_font_size,
            other_font_size=args.other_font_size,
        )
        datasets: List[DisplacementData] = []
        for name, deltak_path_string, r_path_string in args.input:
            deltak_path = Path(deltak_path_string).expanduser().resolve()
            r_path = Path(r_path_string).expanduser().resolve()
            reference, fitted = _load_pair(
                deltak_path,
                r_path,
            )
            item = compute_displacements(
                name=name,
                deltak_path=deltak_path,
                r_path=r_path,
                reference=reference,
                fitted=fitted,
                zero_norm_epsilon=args.zero_norm_epsilon,
            )
            datasets.append(item)
            print(f"Displacement summary [{name}]", item.metrics)

        output = args.output.expanduser().resolve()
        layout = args.layout or (len(datasets), 1)
        png_output, pdf_output = draw(
            datasets,
            output=output,
            levels=sorted(set(args.levels)),
            axis_max=args.axis_max,
            axis_quantile=args.axis_quantile,
            max_scatter_points=args.max_scatter_points,
            random_state=args.random_state,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            dpi=args.dpi,
            layout=layout,
            title_font_size=args.title_font_size,
            axis_font_size=args.axis_font_size,
            legend_font_size=args.legend_font_size,
            other_font_size=args.other_font_size,
        )
        summary_output = (
            args.summary_output.expanduser().resolve()
            if args.summary_output is not None
            else png_output.with_name(f"{png_output.stem}.summary.json")
        )
        payload = {
            "definition": {
                "reference": "r.npz['r']",
                "fitted": "deltak.npz['deltak']",
                "parallel": "<delta_k-r,r>/||r||^2",
                "orthogonal": "||(delta_k-r)-parallel*r||/||r||",
                "per_vector_relative_error": "||delta_k-r||/||r||",
                "identity": "relative_error^2 = parallel^2 + orthogonal^2",
                "batch_relative_frobenius_error": (
                    "||DeltaK-R||_F/||R||_F"
                ),
            },
            "figure": str(png_output),
            "figure_pdf": str(pdf_output),
            "layout": {"rows": layout[0], "columns": layout[1]},
            "font": {
                "family": "Times New Roman",
                "title_size": args.title_font_size,
                "axis_size": args.axis_font_size,
                "legend_size": args.legend_font_size,
                "other_size": args.other_font_size,
            },
            "inputs": [
                {
                    "name": item.name,
                    "deltak_path": str(item.deltak_path),
                    "r_path": str(item.r_path),
                    "metrics": item.metrics,
                }
                for item in datasets
            ],
        }
        _atomic_json(summary_output, payload)
        print(f"Saved displacement summary: {summary_output}")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
