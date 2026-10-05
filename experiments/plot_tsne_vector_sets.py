#!/usr/bin/env python3
"""Plot one or more vector-set pairs with one shared t-SNE embedding.

The figure contains a joint scatter plot, marginal kernel-density estimates,
and optional confidence ellipses.  When multiple ``--pair`` arguments are
provided, every vector from every pair is embedded by a *single* t-SNE fit and
all panels use the same x/y limits.  This is the mode to use when comparing
Hetionet subsets or other datasets.

Examples
--------
Single pair, vectors stored as rows::

    python experiments/plot_tsne_vector_sets.py \
        --a vectors_a.npy --b vectors_b.npy \
        --output figures/a_vs_b.png

Single pair, vectors stored as columns in NPZ files::

    python experiments/plot_tsne_vector_sets.py \
        --a residuals.npz --a-key R \
        --b fitted.npz --b-key delta_k \
        --samples-axis columns \
        --label-a R --label-b 'Delta K' \
        --output figures/residual_fit.png

Several comparable panels using one joint t-SNE fit::

    python experiments/plot_tsne_vector_sets.py \
        --pair hetionet_1 h1_r.npy h1_delta_k.npy \
        --pair hetionet_2 h2_r.npy h2_delta_k.npy \
        --pair hetionet_3 h3_r.npy h3_delta_k.npy \
        --samples-axis rows --columns 3 \
        --output figures/hetionet_shared_tsne.png
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from scipy.stats import chi2, gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


COLOR_A = "#19BFE5"
COLOR_B = "#F27D72"


@dataclass
class PairData:
    name: str
    a: np.ndarray
    b: np.ndarray
    a_indices: np.ndarray
    b_indices: np.ndarray
    embedded_a: Optional[np.ndarray] = None
    embedded_b: Optional[np.ndarray] = None


def _extract_array(value, key: Optional[str], path: Path) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch exists in the project env.
        torch = None

    if key is not None:
        if not isinstance(value, dict):
            raise ValueError(
                f"--key was provided for {path}, but the loaded object is not a mapping"
            )
        if key not in value:
            raise KeyError(
                f"Key {key!r} is absent from {path}; available keys: {sorted(value)}"
            )
        value = value[key]
    elif isinstance(value, dict):
        candidates = {
            name: item
            for name, item in value.items()
            if isinstance(item, np.ndarray)
            or (torch is not None and torch.is_tensor(item))
        }
        if len(candidates) != 1:
            raise ValueError(
                f"{path} contains {len(candidates)} array-like entries; "
                f"select one with --a-key/--b-key. Candidates: {sorted(candidates)}"
            )
        value = next(iter(candidates.values()))

    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D vector matrix in {path}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected numeric vectors in {path}, got dtype={array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        bad = int(array.size - np.isfinite(array).sum())
        raise ValueError(f"{path} contains {bad} non-finite values")
    return array


def load_vector_matrix(path_string: str, key: Optional[str]) -> np.ndarray:
    path = Path(path_string).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Vector file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _extract_array(np.load(path, allow_pickle=False), key, path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            mapping = {name: archive[name] for name in archive.files}
        return _extract_array(mapping, key, path)
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Loading .pt/.pth files requires PyTorch") from exc
        value = torch.load(path, map_location="cpu")
        return _extract_array(value, key, path)
    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t" if suffix == ".tsv" else None
        return _extract_array(np.loadtxt(path, delimiter=delimiter), key, path)
    raise ValueError(
        f"Unsupported vector file {path}; use .npy, .npz, .pt, .pth, .csv, .tsv, or .txt"
    )


def orient_vectors(array: np.ndarray, samples_axis: str, source: str) -> np.ndarray:
    if samples_axis == "rows":
        oriented = array
    elif samples_axis == "columns":
        oriented = array.T
    elif samples_axis == "auto":
        sample_axis = int(array.shape[0] > array.shape[1])
        oriented = array if sample_axis == 0 else array.T
        print(
            f"Auto orientation for {source}: input={tuple(array.shape)}, "
            f"samples_axis={'rows' if sample_axis == 0 else 'columns'}"
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported samples_axis={samples_axis!r}")
    if oriented.shape[0] < 2:
        raise ValueError(f"{source} must contain at least two vectors")
    if oriented.shape[1] < 1:
        raise ValueError(f"{source} vectors must have at least one feature")
    return np.ascontiguousarray(oriented, dtype=np.float32)


def _subsample_pair(
    a: np.ndarray,
    b: np.ndarray,
    maximum: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_indices = np.arange(a.shape[0])
    b_indices = np.arange(b.shape[0])
    if maximum <= 0:
        return a, b, a_indices, b_indices
    if a.shape[0] == b.shape[0]:
        indices = rng.choice(a.shape[0], size=min(maximum, a.shape[0]), replace=False)
        indices.sort()
        return a[indices], b[indices], indices, indices.copy()
    if a.shape[0] > maximum:
        a_indices = np.sort(rng.choice(a.shape[0], size=maximum, replace=False))
        a = a[a_indices]
    if b.shape[0] > maximum:
        b_indices = np.sort(rng.choice(b.shape[0], size=maximum, replace=False))
        b = b[b_indices]
    return a, b, a_indices, b_indices


def load_pairs(args: argparse.Namespace) -> List[PairData]:
    if args.pair:
        if args.a is not None or args.b is not None:
            raise ValueError("Use either --a/--b or repeated --pair, not both")
        specs = args.pair
    else:
        if args.a is None or args.b is None:
            raise ValueError("Provide both --a and --b, or at least one --pair")
        specs = [(args.name, args.a, args.b)]

    names = [spec[0] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Pair names must be unique, got {names}")

    rng = np.random.RandomState(args.random_state)
    pairs = []
    feature_count = None
    for name, a_path, b_path in specs:
        a = orient_vectors(
            load_vector_matrix(a_path, args.a_key), args.samples_axis, a_path
        )
        b = orient_vectors(
            load_vector_matrix(b_path, args.b_key), args.samples_axis, b_path
        )
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"Pair {name!r} has incompatible feature dimensions: "
                f"A={a.shape}, B={b.shape}"
            )
        if feature_count is None:
            feature_count = a.shape[1]
        elif a.shape[1] != feature_count:
            raise ValueError(
                f"All pairs must share one feature space; pair {name!r} has "
                f"{a.shape[1]} features instead of {feature_count}"
            )
        a, b, a_indices, b_indices = _subsample_pair(
            a, b, args.max_points_per_set, rng
        )
        pairs.append(PairData(name, a, b, a_indices, b_indices))
        print(f"Loaded {name}: A={a.shape}, B={b.shape}")
    return pairs


def fit_shared_tsne(
    pairs: Sequence[PairData], args: argparse.Namespace
) -> Tuple[np.ndarray, Optional[PCA]]:
    matrices = []
    slices = []
    offset = 0
    for pair in pairs:
        matrices.extend([pair.a, pair.b])
        a_slice = slice(offset, offset + pair.a.shape[0])
        offset = a_slice.stop
        b_slice = slice(offset, offset + pair.b.shape[0])
        offset = b_slice.stop
        slices.append((a_slice, b_slice))
    x = np.concatenate(matrices, axis=0)

    if args.normalize == "l2":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, np.finfo(np.float32).eps)
    elif args.normalize == "standardize":
        x = StandardScaler(copy=True).fit_transform(x).astype(np.float32, copy=False)

    pca = None
    if args.pca_dim > 0:
        effective_dim = min(args.pca_dim, x.shape[1], x.shape[0] - 1)
        if effective_dim < x.shape[1]:
            pca = PCA(n_components=effective_dim, random_state=args.random_state)
            x = pca.fit_transform(x).astype(np.float32, copy=False)
            explained = float(pca.explained_variance_ratio_.sum())
            print(
                f"Global PCA: {feature_count(pairs)} -> {effective_dim} dimensions, "
                f"explained_variance={explained:.6f}"
            )

    if args.perplexity <= 0 or args.perplexity >= x.shape[0]:
        raise ValueError(
            f"perplexity must be in (0, total_points={x.shape[0]}), "
            f"got {args.perplexity}"
        )
    print(
        f"Fitting one shared t-SNE on {x.shape[0]} points: "
        f"perplexity={args.perplexity}, n_iter={args.n_iter}, "
        f"random_state={args.random_state}"
    )
    embedding = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init="pca",
        metric=args.metric,
        random_state=args.random_state,
        method="barnes_hut",
    ).fit_transform(x)

    for pair, (a_slice, b_slice) in zip(pairs, slices):
        pair.embedded_a = embedding[a_slice]
        pair.embedded_b = embedding[b_slice]
    return embedding, pca


def feature_count(pairs: Sequence[PairData]) -> int:
    return int(pairs[0].a.shape[1])


def confidence_ellipse(
    points: np.ndarray,
    confidence: float,
    edgecolor: str,
) -> Optional[Ellipse]:
    if points.shape[0] < 3:
        return None
    covariance = np.cov(points, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scale = math.sqrt(float(chi2.ppf(confidence, df=2)))
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return Ellipse(
        xy=points.mean(axis=0),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        alpha=0.9,
    )


def padded_limits(values: np.ndarray, fraction: float = 0.06) -> Tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    width = high - low
    if width <= 0:
        width = max(abs(low), 1.0)
    padding = width * fraction
    return low - padding, high + padding


def kde_values(values: np.ndarray, grid: np.ndarray) -> Optional[np.ndarray]:
    if values.size < 2 or float(np.std(values)) <= np.finfo(np.float32).eps:
        return None
    try:
        density = gaussian_kde(values)(grid)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(density).all():
        return None
    return density


def _draw_marginal_x(ax, points_a, points_b, x_grid):
    for values, color in ((points_a[:, 0], COLOR_A), (points_b[:, 0], COLOR_B)):
        density = kde_values(values, x_grid)
        if density is not None:
            ax.fill_between(x_grid, 0, density, color=color, alpha=0.32)
            ax.plot(x_grid, density, color=color, linewidth=1.4)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(False)


def _draw_marginal_y(ax, points_a, points_b, y_grid):
    for values, color in ((points_a[:, 1], COLOR_A), (points_b[:, 1], COLOR_B)):
        density = kde_values(values, y_grid)
        if density is not None:
            ax.fill_betweenx(y_grid, 0, density, color=color, alpha=0.32)
            ax.plot(density, y_grid, color=color, linewidth=1.4)
    ax.set_ylim(y_grid[0], y_grid[-1])
    ax.tick_params(axis="y", labelleft=False, left=False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.grid(False)


def draw_figure(pairs: Sequence[PairData], args: argparse.Namespace) -> None:
    all_points = np.concatenate(
        [points for pair in pairs for points in (pair.embedded_a, pair.embedded_b)],
        axis=0,
    )
    x_limits = padded_limits(all_points[:, 0])
    y_limits = padded_limits(all_points[:, 1])
    shared_span = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    x_center = 0.5 * (x_limits[0] + x_limits[1])
    y_center = 0.5 * (y_limits[0] + y_limits[1])
    x_limits = (x_center - 0.5 * shared_span, x_center + 0.5 * shared_span)
    y_limits = (y_center - 0.5 * shared_span, y_center + 0.5 * shared_span)
    x_grid = np.linspace(*x_limits, 300)
    y_grid = np.linspace(*y_limits, 300)

    columns = max(1, min(args.columns, len(pairs)))
    rows = int(math.ceil(len(pairs) / columns))
    fig = plt.figure(figsize=(5.4 * columns, 5.2 * rows), constrained_layout=False)
    outer = fig.add_gridspec(
        rows,
        columns,
        left=0.07,
        right=0.97,
        bottom=0.08,
        top=0.88,
        wspace=0.22,
        hspace=0.28,
    )

    for index, pair in enumerate(pairs):
        row, column = divmod(index, columns)
        inner = outer[row, column].subgridspec(
            2,
            2,
            width_ratios=(4.0, 1.0),
            height_ratios=(1.0, 4.0),
            hspace=0.04,
            wspace=0.04,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_top = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax_joint)
        ax_empty = fig.add_subplot(inner[0, 1])
        ax_empty.axis("off")

        if args.draw_pairs and pair.embedded_a.shape[0] == pair.embedded_b.shape[0]:
            segments = np.stack([pair.embedded_a, pair.embedded_b], axis=1)
            ax_joint.add_collection(
                LineCollection(
                    segments,
                    colors="#808080",
                    linewidths=0.45,
                    alpha=0.18,
                    zorder=1,
                )
            )
        ax_joint.scatter(
            pair.embedded_a[:, 0],
            pair.embedded_a[:, 1],
            s=args.point_size,
            c=COLOR_A,
            alpha=args.point_alpha,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        ax_joint.scatter(
            pair.embedded_b[:, 0],
            pair.embedded_b[:, 1],
            s=args.point_size,
            c=COLOR_B,
            alpha=args.point_alpha,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        if args.confidence > 0:
            for points, color in (
                (pair.embedded_a, COLOR_A),
                (pair.embedded_b, COLOR_B),
            ):
                ellipse = confidence_ellipse(points, args.confidence, color)
                if ellipse is not None:
                    ax_joint.add_patch(ellipse)

        ax_joint.set_xlim(x_limits)
        ax_joint.set_ylim(y_limits)
        ax_joint.set_xlabel("t-SNE 1")
        ax_joint.set_ylabel("t-SNE 2")
        ax_joint.grid(True, color="#D8D8D8", linewidth=0.5, alpha=0.6)
        ax_joint.set_axisbelow(True)
        ax_joint.set_aspect("equal", adjustable="box")
        ax_top.set_title(
            f"{pair.name}  (A={pair.embedded_a.shape[0]}, B={pair.embedded_b.shape[0]})",
            fontsize=11,
            pad=4,
        )
        _draw_marginal_x(ax_top, pair.embedded_a, pair.embedded_b, x_grid)
        _draw_marginal_y(ax_right, pair.embedded_a, pair.embedded_b, y_grid)

    for index in range(len(pairs), rows * columns):
        row, column = divmod(index, columns)
        empty = fig.add_subplot(outer[row, column])
        empty.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLOR_A,
            markeredgecolor="none",
            markersize=7,
            label=args.label_a,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLOR_B,
            markeredgecolor="none",
            markersize=7,
            label=args.label_b,
        ),
    ]
    if args.title:
        fig.suptitle(args.title, fontsize=14, y=0.975)
        legend_y = 0.94
    else:
        legend_y = 0.965
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        frameon=False,
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output}")


def _safe_key(name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return key or "pair"


def save_embedding(pairs: Sequence[PairData], output_string: str) -> None:
    output = Path(output_string).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    for pair in pairs:
        prefix = _safe_key(pair.name)
        payload[f"{prefix}__A"] = pair.embedded_a
        payload[f"{prefix}__B"] = pair.embedded_b
        payload[f"{prefix}__A_indices"] = pair.a_indices
        payload[f"{prefix}__B_indices"] = pair.b_indices
    np.savez_compressed(output, **payload)
    print(f"Saved embedding: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw A/B vector sets as joint t-SNE scatter plots with marginal "
            "KDEs. Repeated --pair inputs share one t-SNE fit and axis scale."
        )
    )
    parser.add_argument("--a", help="Vector file for A in single-pair mode")
    parser.add_argument("--b", help="Vector file for B in single-pair mode")
    parser.add_argument("--name", default="A vs B", help="Single-pair panel name")
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("NAME", "A_PATH", "B_PATH"),
        help="Add a named A/B pair; repeat to create shared-embedding panels",
    )
    parser.add_argument("--a-key", help="Array key for every A NPZ/PT mapping")
    parser.add_argument("--b-key", help="Array key for every B NPZ/PT mapping")
    parser.add_argument(
        "--samples-axis",
        choices=("rows", "columns", "auto"),
        default="rows",
        help=(
            "Where samples are stored. 'columns' converts [D,N] to [N,D]; "
            "'auto' treats the smaller dimension as samples (default: rows)."
        ),
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "l2", "standardize"),
        default="none",
        help="One global preprocessing rule applied before PCA/t-SNE",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="Global PCA dimension before t-SNE; 0 disables PCA (default: 50)",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=200.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-points-per-set",
        type=int,
        default=0,
        help="Optional deterministic subsample limit; 0 keeps all vectors",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence ellipse mass in (0,1); 0 hides ellipses",
    )
    parser.add_argument("--draw-pairs", action="store_true")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--title")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--point-alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--embedding-output",
        help="Optional NPZ path for the shared two-dimensional coordinates",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.pca_dim < 0:
        raise ValueError("--pca-dim must be non-negative")
    if args.n_iter < 250:
        raise ValueError("--n-iter must be at least 250 for sklearn t-SNE")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.max_points_per_set < 0:
        raise ValueError("--max-points-per-set must be non-negative")
    if not (args.confidence == 0 or 0 < args.confidence < 1):
        raise ValueError("--confidence must be 0 or lie in (0,1)")
    if args.columns <= 0:
        raise ValueError("--columns must be positive")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if not (0 < args.point_alpha <= 1):
        raise ValueError("--point-alpha must lie in (0,1]")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        pairs = load_pairs(args)
        fit_shared_tsne(pairs, args)
        draw_figure(pairs, args)
        if args.embedding_output:
            save_embedding(pairs, args.embedding_output)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
