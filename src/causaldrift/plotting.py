"""Plot evaluation summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .evaluation import METRICS


METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "norm_shd": "1 − normalized SHD",
    "norm_whd": "1 − normalized weighted Hamming",
}
ERROR_METRICS = {"norm_shd", "norm_whd"}

# Trade-off plot colors and markers.
TRADEOFF_STYLES = {
    "acd": ("#0072B2", "o"),
    "cuts": ("#E69F00", "s"),
    "grasp": ("#009E73", "^"),
    "ngc": ("#D55E00", "D"),
    "uncle": ("#CC79A7", "P"),
    "varlingam": ("#6A3D9A", "X"),
    "cdans": ("#F0E442", "v"),
    "fpcmci": ("#56B4E9", "<"),
    "kausal": ("#7F7F7F", "h"),
    "pcmci+": ("#A65628", ">"),
}
_FALLBACK_TRADEOFF_COLORS = ("#1B9E77", "#E7298A", "#7570B3", "#66A61E")
_FALLBACK_TRADEOFF_MARKERS = ("8", "p", "*", "d")


def _read_summary(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        source = source / "summary_metrics.csv"
    if not source.exists():
        raise FileNotFoundError(f"No summary_metrics.csv at {source}")
    return pd.read_csv(source)


def aggregate_for_plots(summary: pd.DataFrame) -> pd.DataFrame:
    """Average dataset-level summary rows."""

    required = {"model", "model_label"} | {f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for (model, label), group in summary.groupby(["model", "model_label"], sort=False):
        row: dict[str, object] = {"model": model, "model_label": label, "dataset_count": len(group)}
        for metric in METRICS:
            mean = float(pd.to_numeric(group[f"{metric}_mean"], errors="coerce").mean())
            std = float(pd.to_numeric(group[f"{metric}_std"], errors="coerce").mean())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            utility = (1.0 - mean if metric in ERROR_METRICS else mean)
            row[f"{metric}_utility"] = utility
            row[f"{metric}_mu_minus_sigma"] = utility - std
        rows.append(row)
    output = pd.DataFrame.from_records(rows)
    output["balanced_mu_minus_sigma"] = output[[f"{metric}_mu_minus_sigma" for metric in METRICS]].mean(axis=1)
    return output


def _tradeoff_style_map(models: pd.Series) -> dict[str, tuple[str, str]]:
    """Return one stable, visually distinct style per trade-off method."""

    styles: dict[str, tuple[str, str]] = {}
    fallback_index = 0
    for model in dict.fromkeys(models.astype(str)):
        if model in TRADEOFF_STYLES:
            styles[model] = TRADEOFF_STYLES[model]
            continue
        styles[model] = (
            _FALLBACK_TRADEOFF_COLORS[fallback_index % len(_FALLBACK_TRADEOFF_COLORS)],
            _FALLBACK_TRADEOFF_MARKERS[fallback_index % len(_FALLBACK_TRADEOFF_MARKERS)],
        )
        fallback_index += 1
    return styles


def plot_tradeoff(aggregate: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot accuracy/robustness trade-offs: high and right is better."""

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = _tradeoff_style_map(aggregate["model"])
    labels = dict(zip(aggregate["model"].astype(str), aggregate["model_label"].astype(str)))
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for metric, axis in zip(METRICS, axes.flat):
        x = aggregate[f"{metric}_std"]
        y = aggregate[f"{metric}_utility"]
        for model, x_value, y_value in zip(aggregate["model"].astype(str), x, y):
            if not (np.isfinite(x_value) and np.isfinite(y_value)):
                continue
            color, marker = styles[model]
            axis.scatter(
                x_value,
                y_value,
                s=88,
                color=color,
                marker=marker,
                edgecolors="#1A1A1A",
                linewidths=0.65,
                alpha=0.96,
                zorder=3,
            )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Window standard deviation (lower is better)")
        axis.set_ylabel("Mean performance (higher is better)")
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
        axis.set_ylim(-0.03, 1.03)
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=styles[model][1],
            markerfacecolor=styles[model][0],
            markeredgecolor="#1A1A1A",
            markersize=8,
            label=labels[model],
        )
        for model in styles
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False)
    figure.suptitle("CausalDrift accuracy–robustness trade-off", fontsize=14)
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def plot_ranking(aggregate: pd.DataFrame, out_path: str | Path) -> Path:
    """Rank models by μ−σ after making all four metrics higher-is-better."""

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    panels = [*METRICS, "balanced"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for panel, axis in zip(panels, axes.flat):
        column = "balanced_mu_minus_sigma" if panel == "balanced" else f"{panel}_mu_minus_sigma"
        ordered = aggregate.sort_values(column, ascending=True, na_position="last")
        colors = ["#8b9eb7" if name != ordered.iloc[-1]["model"] else "#c75b39" for name in ordered["model"]]
        axis.barh(ordered["model_label"], ordered[column], color=colors)
        axis.set_title("Balanced μ − σ" if panel == "balanced" else METRIC_LABELS[panel])
        axis.set_xlabel("μ − σ")
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlim(min(-0.05, float(np.nanmin(ordered[column])) - 0.03), 1.03)
    axes.flat[-1].axis("off")
    figure.suptitle("CausalDrift ranking from random-window mean and standard deviation", fontsize=14)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def make_plots(summary_path: str | Path, out_dir: str | Path) -> tuple[pd.DataFrame, Path, Path]:
    """Create trade-off and μ−σ ranking plots."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_for_plots(_read_summary(summary_path))
    aggregate.to_csv(destination / "plot_summary.csv", index=False)
    tradeoff = plot_tradeoff(aggregate, destination / "tradeoff.png")
    ranking = plot_ranking(aggregate, destination / "ranking_mu_minus_sigma.png")
    return aggregate, tradeoff, ranking
