#!/usr/bin/env python3
"""Plot saved off-subject-token disturbances without simulated data.

For each saved token, read ``d_j = Delta h_j^o`` and ``y_j = W0 h_j^o``
from ``rewrite_prompt_off_tokens/part_*.npz`` and plot

    a_j = <d_j, y_j> / ||y_j||_2^2,
    b_j = ||d_j - a_j y_j||_2 / ||y_j||_2.

Thus ``sqrt(a_j^2+b_j^2) = ||d_j||_2/||y_j||_2`` exactly.  The script
uses neither t-SNE/PCA nor generated vectors.  It streams all part files;
full-data metrics use every token, while a bounded uniform sample is retained
for drawing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager

for _font_dir in (
    Path("/usr/share/fonts/truetype/msttcorefonts"),
    Path("C:/Windows/Fonts"),
):
    if _font_dir.is_dir():
        for _pattern in ("Times_New_Roman*.ttf", "times*.ttf"):
            for _font_path in _font_dir.glob(_pattern):
                try:
                    font_manager.fontManager.addfont(str(_font_path))
                except RuntimeError:
                    pass

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm


@dataclass
class SavedDisturbance:
    name: str
    source: Path
    part_files: List[Path]
    parallel: np.ndarray
    orthogonal: np.ndarray
    relative: np.ndarray
    metrics: Dict[str, object]


class Reservoir:
    """Uniform sample obtained by retaining the smallest random priorities."""

    def __init__(self, maximum: int, seed: int) -> None:
        self.maximum = maximum
        self.rng = np.random.default_rng(seed)
        self.priority = np.empty(0, dtype=np.float64)
        self.a = np.empty(0, dtype=np.float64)
        self.b = np.empty(0, dtype=np.float64)
        self.e = np.empty(0, dtype=np.float64)

    def add(self, a: np.ndarray, b: np.ndarray, e: np.ndarray) -> None:
        if a.size == 0:
            return
        priority = self.rng.random(a.size)
        priority = np.concatenate((self.priority, priority))
        a = np.concatenate((self.a, a))
        b = np.concatenate((self.b, b))
        e = np.concatenate((self.e, e))
        if priority.size > self.maximum:
            keep = np.argpartition(priority, self.maximum - 1)[: self.maximum]
            priority, a, b, e = (
                priority[keep],
                a[keep],
                b[keep],
                e[keep],
            )
        self.priority, self.a, self.b, self.e = priority, a, b, e

    def values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(self.priority)
        return self.a[order], self.b[order], self.e[order]


def discover_parts(source: Path) -> List[Path]:
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Input file is not NPZ: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Saved-data path does not exist: {source}")
    files = sorted(source.glob("part_*.npz"))
    if not files:
        files = sorted(
            (source / "rewrite_prompt_off_tokens").glob("part_*.npz")
        )
    if not files:
        raise FileNotFoundError(
            f"No part_*.npz found in {source} or its "
            "rewrite_prompt_off_tokens child"
        )
    return files


def read_part(
    path: Path, delta_key: str, reference_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [
            key
            for key in (delta_key, reference_key)
            if key not in archive.files
        ]
        if missing:
            raise KeyError(
                f"{path} is missing {missing}; keys={archive.files}. "
                "Old caches without W0H must be regenerated with "
                "last_layer_fit_error=on."
            )
        delta = np.asarray(archive[delta_key], dtype=np.float64)
        reference = np.asarray(archive[reference_key], dtype=np.float64)
    if delta.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            f"{path}: arrays must be rank 2; got {delta.shape}, "
            f"{reference.shape}"
        )
    if delta.shape != reference.shape:
        raise ValueError(
            f"{path}: shape mismatch {delta_key}={delta.shape}, "
            f"{reference_key}={reference.shape}"
        )
    for key, value in ((delta_key, delta), (reference_key, reference)):
        if not np.isfinite(value).all():
            raise ValueError(f"{path}:{key} contains non-finite values")
    return delta, reference


def compute_dataset(
    *,
    name: str,
    source: Path,
    delta_key: str,
    reference_key: str,
    zero_norm_epsilon: float,
    max_plot_points: int,
    seed: int,
) -> SavedDisturbance:
    part_files = discover_parts(source)
    reservoir = Reservoir(max_plot_points, seed)
    total = valid_total = zero_total = 0
    dimension = None
    delta_fro_sq = reference_fro_sq = 0.0
    e_sum = a_sum = abs_a_sum = b_sum = 0.0
    e_max = identity_max = 0.0

    for path in part_files:
        delta, reference = read_part(path, delta_key, reference_key)
        if dimension is None:
            dimension = int(delta.shape[1])
        elif delta.shape[1] != dimension:
            raise ValueError(f"Feature dimension changes in {path}")

        reference_sq = np.einsum(
            "ij,ij->i", reference, reference, dtype=np.float64
        )
        delta_sq = np.einsum(
            "ij,ij->i", delta, delta, dtype=np.float64
        )
        dot = np.einsum(
            "ij,ij->i", delta, reference, dtype=np.float64
        )
        valid = reference_sq > zero_norm_epsilon**2
        total += int(valid.size)
        valid_total += int(valid.sum())
        zero_total += int((~valid).sum())
        delta_fro_sq += float(delta_sq.sum(dtype=np.float64))
        reference_fro_sq += float(reference_sq.sum(dtype=np.float64))
        if not valid.any():
            continue

        r2, d2, product = reference_sq[valid], delta_sq[valid], dot[valid]
        a = product / r2
        b2 = (d2 - np.square(product) / r2) / r2
        b = np.sqrt(np.maximum(b2, 0.0))
        e = np.sqrt(np.maximum(d2 / r2, 0.0))
        reconstructed = np.sqrt(np.square(a) + np.square(b))
        identity_max = max(
            identity_max,
            float(np.max(np.abs(e - reconstructed), initial=0.0)),
        )
        e_sum += float(e.sum(dtype=np.float64))
        a_sum += float(a.sum(dtype=np.float64))
        abs_a_sum += float(np.abs(a).sum(dtype=np.float64))
        b_sum += float(b.sum(dtype=np.float64))
        e_max = max(e_max, float(e.max(initial=0.0)))
        reservoir.add(a, b, e)

    if valid_total == 0 or reference_fro_sq <= 0.0:
        raise ValueError(f"No usable nonzero W0H vectors for {name}")
    a, b, e = reservoir.values()
    metrics: Dict[str, object] = {
        "num_part_files": len(part_files),
        "num_tokens_total": total,
        "num_tokens_valid": valid_total,
        "num_zero_norm_reference_tokens_excluded": zero_total,
        "dimension": dimension,
        "delta_h_off_fro_sq": delta_fro_sq,
        "w0_h_off_fro_sq": reference_fro_sq,
        "global_frobenius_ratio": math.sqrt(
            delta_fro_sq / reference_fro_sq
        ),
        "mean_relative_token_disturbance": e_sum / valid_total,
        "mean_parallel_relative_disturbance": a_sum / valid_total,
        "mean_absolute_parallel_relative_disturbance": (
            abs_a_sum / valid_total
        ),
        "mean_orthogonal_relative_disturbance": b_sum / valid_total,
        "max_relative_token_disturbance": e_max,
        "decomposition_identity_max_abs_error": identity_max,
        "plot_sample_size": int(e.size),
        "plot_sample_is_full_data": bool(e.size == valid_total),
        "plot_sample_relative_quantiles": {
            "median": float(np.quantile(e, 0.50)),
            "p90": float(np.quantile(e, 0.90)),
            "p95": float(np.quantile(e, 0.95)),
            "p99": float(np.quantile(e, 0.99)),
        },
    }
    return SavedDisturbance(
        name=name,
        source=source,
        part_files=part_files,
        parallel=a,
        orthogonal=b,
        relative=e,
        metrics=metrics,
    )


def parse_layout(value: str) -> Tuple[int, int]:
    value = value.strip().strip("()")
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("layout must be ROWS,COLUMNS")
    try:
        rows, columns = map(int, parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layout values must be integers"
        ) from error
    if rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError("layout values must be positive")
    return rows, columns


def configure_style(args: argparse.Namespace) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": args.other_font_size,
            "axes.titlesize": args.title_font_size,
            "axes.titleweight": "bold",
            "axes.labelsize": args.axis_font_size,
            "axes.labelweight": "bold",
            "xtick.labelsize": args.axis_font_size,
            "ytick.labelsize": args.axis_font_size,
            "legend.fontsize": args.legend_font_size,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
        }
    )


def draw(
    datasets: Sequence[SavedDisturbance], args: argparse.Namespace
) -> Dict[str, Path]:
    rows, columns = args.layout or (len(datasets), 1)
    if args.axis_max > 0.0:
        radius = args.axis_max
    else:
        errors = np.concatenate([item.relative for item in datasets])
        radius = float(np.quantile(errors, args.axis_quantile))
        radius = max(radius, max(args.levels, default=0.0)) * 1.08
    radius = max(radius, np.finfo(np.float64).eps)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.35 * columns, 2.55 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    occupied_axes = flat_axes[: len(datasets)]
    theta = np.linspace(0.0, math.pi, 400)
    hexbins = []
    maximum_count = 1.0

    for item, axis in zip(datasets, occupied_axes):
        hexbin = axis.hexbin(
            item.parallel,
            item.orthogonal,
            gridsize=args.hexbin_gridsize,
            extent=(-radius, radius, 0.0, radius),
            mincnt=1,
            cmap="viridis",
            linewidths=0.0,
            rasterized=True,
        )
        hexbins.append(hexbin)
        if hexbin.get_array().size:
            maximum_count = max(
                maximum_count, float(hexbin.get_array().max())
            )
        for level in sorted(set(args.levels)):
            axis.plot(
                level * np.cos(theta),
                level * np.sin(theta),
                color="#777777",
                linewidth=0.75,
                linestyle=(0, (4, 3)),
            )
            axis.text(
                level / math.sqrt(2.0),
                level / math.sqrt(2.0),
                f"{level:g}",
                fontsize=args.other_font_size,
                color="#555555",
            )
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlim(-radius, radius)
        axis.set_ylim(0.0, radius)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D8D8D8", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
        axis.set_title(item.name, fontweight="bold")
        axis.set_xlabel(
            r"Parallel relative disturbance $a_j$", fontweight="bold"
        )
        axis.set_ylabel(
            r"Orthogonal relative disturbance $b_j$", fontweight="bold"
        )
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick.set_fontweight("bold")
        axis.text(
            0.97,
            0.95,
            (
                rf"$\mathrm{{mean}}(e_j)="
                f"{item.metrics['mean_relative_token_disturbance']:.4f}$"
                "\n"
                rf"$\epsilon_F="
                f"{item.metrics['global_frobenius_ratio']:.4f}$"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=args.other_font_size,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "pad": 1.5,
            },
        )

    for axis in flat_axes[len(datasets) :]:
        axis.set_axis_off()

    shared_norm = LogNorm(vmin=1.0, vmax=max(2.0, maximum_count))
    for hexbin in hexbins:
        hexbin.set_norm(shared_norm)
    mappable = ScalarMappable(norm=shared_norm, cmap="viridis")
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable, ax=occupied_axes, fraction=0.026, pad=0.02
    )
    colorbar.set_label("Saved-token density (log count)", fontweight="bold")
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    stem = args.output.expanduser().resolve().with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for file_format in args.formats:
        path = stem.with_suffix(f".{file_format}")
        figure.savefig(path, bbox_inches="tight", pad_inches=0.04)
        outputs[file_format] = path
        print(f"Saved off-token figure: {path}")
    plt.close(figure)
    return outputs


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Delta H_off from saved rewrite_prompt_off_tokens parts."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("NAME", "SAVED_PATH"),
        required=True,
        help=(
            "Repeatable named input. SAVED_PATH is a part NPZ, the "
            "rewrite_prompt_off_tokens directory, or its parent directory."
        ),
    )
    parser.add_argument("--delta-key", default="delta_h_off")
    parser.add_argument(
        "--reference-key",
        default="w0_h_off",
        help=(
            "Use w0_h_off (default) or w0_h_off_original_model."
        ),
    )
    parser.add_argument("--layout", type=parse_layout)
    parser.add_argument("--zero-norm-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--levels", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.5]
    )
    parser.add_argument("--axis-max", type=float, default=0.0)
    parser.add_argument("--axis-quantile", type=float, default=0.995)
    parser.add_argument("--max-plot-points", type=int, default=100_000)
    parser.add_argument("--hexbin-gridsize", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--title-font-size", type=float, default=10.0)
    parser.add_argument("--axis-font-size", type=float, default=8.0)
    parser.add_argument("--legend-font-size", type=float, default=8.0)
    parser.add_argument("--other-font-size", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    names = [name for name, _ in args.input]
    if len(names) != len(set(names)):
        raise ValueError(f"Input names must be unique: {names}")
    if args.layout and args.layout[0] * args.layout[1] < len(args.input):
        raise ValueError("layout has fewer panels than inputs")
    if args.zero_norm_epsilon < 0:
        raise ValueError("zero-norm-epsilon must be non-negative")
    if any(level <= 0 for level in args.levels):
        raise ValueError("levels must be positive")
    if args.axis_max < 0 or not 0 < args.axis_quantile <= 1:
        raise ValueError("invalid axis-max or axis-quantile")
    if args.max_plot_points <= 0 or args.hexbin_gridsize <= 1:
        raise ValueError("invalid max-plot-points or hexbin-gridsize")
    for key in (
        "title_font_size",
        "axis_font_size",
        "legend_font_size",
        "other_font_size",
    ):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        configure_style(args)
        datasets: List[SavedDisturbance] = []
        for index, (name, source_text) in enumerate(args.input):
            item = compute_dataset(
                name=name,
                source=Path(source_text).expanduser().resolve(),
                delta_key=args.delta_key,
                reference_key=args.reference_key,
                zero_norm_epsilon=args.zero_norm_epsilon,
                max_plot_points=args.max_plot_points,
                seed=args.random_state + index,
            )
            datasets.append(item)
            print(f"Off-token summary [{name}]", item.metrics)
        outputs = draw(datasets, args)
        stem = args.output.expanduser().resolve().with_suffix("")
        summary_path = (
            args.summary_output.expanduser().resolve()
            if args.summary_output
            else stem.with_suffix(".summary.json")
        )
        atomic_json(
            summary_path,
            {
                "simulation": False,
                "embedding": "none",
                "definition": {
                    "delta": f"part_*.npz[{args.delta_key!r}]",
                    "reference": f"part_*.npz[{args.reference_key!r}]",
                    "parallel": "<Delta h,W0 h>/||W0 h||_2^2",
                    "orthogonal": (
                        "||Delta h-parallel*(W0 h)||_2/||W0 h||_2"
                    ),
                    "relative": "||Delta h||_2/||W0 h||_2",
                    "epsilon_F": "||Delta H||_F/||W0 H||_F",
                },
                "outputs": {key: str(value) for key, value in outputs.items()},
                "inputs": [
                    {
                        "name": item.name,
                        "source": str(item.source),
                        "part_files": [str(path) for path in item.part_files],
                        "metrics": item.metrics,
                    }
                    for item in datasets
                ],
            },
        )
        print(f"Saved off-token summary: {summary_path}")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
