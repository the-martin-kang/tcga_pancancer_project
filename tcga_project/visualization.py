"""Reusable visualization functions for the TCGA Pan-Cancer notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D


def _maybe_savefig(save_path: str | Path | None, dpi: int = 150) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[figure] saved: {save_path}")


TAB20_REORDER = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# Use only filled markers. In v4, the last five labels used line-only markers
# such as "+", "x", "1", "2", "3" together with linewidths=0.0, making
# PRAD/SARC/STAD/THCA/UCEC invisible in both scatter points and legend.
MARKERS = [
    "o", "s", "^", "D", "P", "X", "v", "<", ">", "*",
    "h", "p", "8", "H", "d", "o", "s", "^", "D", "P",
]
def make_label_color_map(labels: Iterable, palette: str | None = None) -> dict[str, str]:
    """Return a stable label-to-color mapping for 10, 20, or more classes."""
    unique_labels = sorted(pd.Series(list(labels)).dropna().astype(str).unique().tolist())
    n = len(unique_labels)
    if n == 0:
        return {}
    if palette is not None:
        cmap = plt.get_cmap(palette, n)
        colors = [to_hex(cmap(i)) for i in range(n)]
    elif n <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [to_hex(cmap(i)) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [to_hex(cmap(i)) for i in TAB20_REORDER[:n]]
    elif n <= 60:
        colors = []
        for cmap_name in ["tab20", "tab20b", "tab20c"]:
            cmap = plt.get_cmap(cmap_name)
            colors.extend([to_hex(cmap(i)) for i in range(20)])
        colors = colors[:n]
    else:
        cmap = plt.get_cmap("nipy_spectral", n)
        colors = [to_hex(cmap(i)) for i in range(n)]
    return dict(zip(unique_labels, colors))

def make_label_marker_map(labels: Iterable) -> dict[str, str]:
    """Return a stable label-to-marker mapping using only filled markers.

    Matplotlib line-only markers such as ``+`` and ``x`` can disappear when
    scatter plots use zero linewidth.  This helper intentionally avoids them,
    so every class has a visible point in both the plot and the legend.
    """
    unique_labels = sorted(pd.Series(list(labels)).dropna().astype(str).unique().tolist())
    return {label: MARKERS[i % len(MARKERS)] for i, label in enumerate(unique_labels)}


def _legend_handles_for_labels(
    label_list: Sequence[str],
    color_map: Mapping[str, str],
    marker_map: Mapping[str, str] | None = None,
    markersize: float = 6.0,
) -> list[Line2D]:
    """Build explicit legend handles instead of relying on scatter handles.

    This is deliberately robust for 20-class plots: even if a scatter marker is
    drawn with small alpha/size, the legend marker remains visible.
    """
    if marker_map is None:
        marker_map = make_label_marker_map(label_list)
    handles: list[Line2D] = []
    for label in label_list:
        color = color_map.get(str(label), "C0")
        marker = marker_map.get(str(label), "o")
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                label=str(label),
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.9,
                color=color,
                markersize=markersize,
            )
        )
    return handles

def plot_palette_preview(labels: Iterable, title: str = "Cancer-type color palette", save_path: str | Path | None = None):
    """Show the categorical color mapping used by embedding plots."""
    color_map = make_label_color_map(labels)
    label_list = sorted(color_map)
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.25 * len(label_list) + 1.0)))
    y_pos = np.arange(len(label_list))
    ax.barh(y_pos, np.ones(len(label_list)), color=[color_map[label] for label in label_list])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_list)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_class_counts(y: pd.Series, title: str = "Class distribution", save_path: str | Path | None = None, color_map: Mapping[str, str] | None = None):
    counts = y.value_counts().sort_index()
    labels = counts.index.astype(str).tolist()
    if color_map is None:
        color_map = make_label_color_map(labels)
    colors = [color_map.get(str(label), "C0") for label in labels]
    fig, ax = plt.subplots(figsize=(max(9, len(counts) * 0.55), 4.8))
    ax.bar(labels, counts.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Cancer type")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_expression_distribution(
    X: pd.DataFrame,
    max_values: int = 200_000,
    random_state: int = 42,
    title: str = "Gene expression value distribution",
    save_path: str | Path | None = None,
):
    rng = np.random.default_rng(random_state)
    values = X.to_numpy().ravel()
    values = values[np.isfinite(values)]
    if values.size > max_values:
        values = rng.choice(values, size=max_values, replace=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=80, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Expression value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_embedding(
    embedding: np.ndarray,
    labels: Iterable,
    title: str,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    save_path: str | Path | None = None,
    alpha: float = 0.75,
    size: float = 14,
    color_map: Mapping[str, str] | None = None,
    use_markers: bool = True,
):
    """Scatter plot for PCA/t-SNE/UMAP embeddings with robust 20-class legends.

    v4/v4.1 used line-only markers for the last five classes, which caused
    PRAD/SARC/STAD/THCA/UCEC to appear without legend markers under some
    matplotlib settings.  This version uses only filled markers and constructs
    explicit ``Line2D`` legend handles, so all labels remain visible.
    """
    labels = pd.Series(list(labels), name="label").astype(str).reset_index(drop=True)
    embedding = np.asarray(embedding)
    unique_labels = sorted(labels.unique())
    if color_map is None:
        color_map = make_label_color_map(unique_labels)
    marker_map = make_label_marker_map(unique_labels)

    fig, ax = plt.subplots(
        figsize=(10.5 if len(unique_labels) >= 16 else 9, 7.2 if len(unique_labels) >= 16 else 6.5)
    )
    for label in unique_labels:
        idx = (labels == label).to_numpy()
        marker = marker_map.get(str(label), "o") if use_markers else "o"
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=size,
            alpha=alpha,
            label=str(label),
            c=[color_map.get(str(label), "C0")],
            marker=marker,
            edgecolors="white",
            linewidths=0.15,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    legend_cols = 1 if len(unique_labels) <= 12 else (2 if len(unique_labels) <= 24 else 3)
    legend_handles = _legend_handles_for_labels(
        unique_labels,
        color_map=color_map,
        marker_map=marker_map if use_markers else {label: "o" for label in unique_labels},
        markersize=6.2,
    )
    ax.legend(
        handles=legend_handles,
        title="Cancer type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        ncol=legend_cols,
        handletextpad=0.7,
        columnspacing=1.0,
        borderaxespad=0.4,
        frameon=True,
    )
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax

def plot_confusion_matrix(
    cm: pd.DataFrame,
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
    normalize: bool = False,
    counts: pd.DataFrame | None = None,
    count_df: pd.DataFrame | None = None,
    annotate_counts: pd.DataFrame | None = None,
    show_counts: bool = True,
    value_format: str | None = None,
    figsize: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    annotate: bool | None = None,
):
    """Plot a confusion matrix with class-imbalance-friendly options.

    For imbalanced multi-class data, pass a row-normalized confusion matrix
    created with ``normalize="true"`` and set ``normalize=True``. Then each
    true class(row) uses the same 0..1 color scale, so large classes such as
    BRCA do not dominate the heatmap merely because they have more samples.

    ``counts`` and ``annotate_counts`` are aliases. If either is supplied, raw
    counts are printed below the normalized percentage in each cell.
    """
    data = cm.values.astype(float)
    n_classes = len(cm.index)
    if figsize is None:
        figsize = (max(8, 0.45 * n_classes + 3.5), max(7, 0.40 * n_classes + 3.0))
    if annotate is None:
        annotate = n_classes <= 15

    fig, ax = plt.subplots(figsize=figsize)
    imshow_kwargs = {"interpolation": "nearest"}
    if normalize:
        imshow_kwargs.update({"vmin": 0.0 if vmin is None else vmin, "vmax": 1.0 if vmax is None else vmax})
    else:
        if vmin is not None:
            imshow_kwargs["vmin"] = vmin
        if vmax is not None:
            imshow_kwargs["vmax"] = vmax
    im = ax.imshow(data, **imshow_kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label or ("Row-normalized rate" if normalize else "Count"))

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_yticklabels(cm.index)

    count_matrix = counts if counts is not None else (count_df if count_df is not None else annotate_counts)
    count_values = None
    if count_matrix is not None:
        count_values = count_matrix.reindex(index=cm.index, columns=cm.columns).values

    if value_format is None:
        value_format = ".0%" if normalize else "d"

    if annotate:
        font_size = 7 if n_classes >= 15 else 8
        threshold = 0.5 if normalize else (np.nanmax(data) / 2 if data.size else 0)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if normalize:
                    text = format(value, value_format)
                else:
                    text = format(int(round(value)), value_format)
                if show_counts and count_values is not None:
                    text = f"{text}\n(n={int(round(count_values[i, j]))})"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="white" if value > threshold else "black",
                )

    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax

def plot_metrics_table(
    metrics_df: pd.DataFrame,
    metric: str = "macro_f1",
    title: str = "Model performance",
    save_path: str | Path | None = None,
):
    df = metrics_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["model"], df[metric])
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_xlim(0, min(1.0, max(0.05, df[metric].max() * 1.10)))
    for i, v in enumerate(df[metric]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_per_class_metric(
    report_df: pd.DataFrame,
    metric: str = "recall",
    title: str = "Per-class performance",
    save_path: str | Path | None = None,
):
    """Plot a per-class metric from a sklearn classification report table."""
    df = report_df.copy()
    df = df[~df["class"].isin(["accuracy", "macro avg", "weighted avg"])]
    df = df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.28)))
    ax.barh(df["class"].astype(str), df[metric])
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(df[metric]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_classification_report_metric(
    report_df: pd.DataFrame,
    metric: str = "recall",
    title: str = "Per-class performance",
    save_path: str | Path | None = None,
):
    """Backward-compatible alias for plot_per_class_metric."""
    return plot_per_class_metric(report_df, metric=metric, title=title, save_path=save_path)


def plot_top_confusion_pairs(
    confusion_pairs_df: pd.DataFrame,
    metric: str = "error_rate_within_true_class",
    title: str = "Top confusion pairs",
    save_path: str | Path | None = None,
):
    if confusion_pairs_df.empty:
        print("No off-diagonal confusion pairs to plot.")
        return None, None
    df = confusion_pairs_df.copy()
    df["pair"] = df["true_class"].astype(str) + " → " + df["predicted_class"].astype(str)
    df = df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.32)))
    ax.barh(df["pair"], df[metric])
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_xlim(0, max(0.05, min(1.0, df[metric].max() * 1.15)))
    for i, row in enumerate(df.itertuples(index=False)):
        value = getattr(row, metric)
        ax.text(value, i, f" {value:.1%} (n={row.count})", va="center", fontsize=8)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_top_genes(
    top_gene_df: pd.DataFrame,
    score_col: str,
    title: str = "Top genes",
    top_n: int = 20,
    save_path: str | Path | None = None,
):
    df = top_gene_df.head(top_n).copy()
    if "gene" not in df.columns:
        raise ValueError("top_gene_df must contain a 'gene' column.")
    df = df.sort_values(score_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    ax.barh(df["gene"].astype(str), df[score_col])
    ax.set_title(title)
    ax.set_xlabel(score_col)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_training_history(history: pd.DataFrame, save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_title("MLP training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax


def plot_feature_set_comparison(
    comparison_df: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "balanced_accuracy", "macro_f1"),
    title: str = "Feature set size comparison",
    save_path: str | Path | None = None,
):
    """Grouped bar plot for feature-set comparison results."""
    if comparison_df.empty:
        print("No feature-set comparison results to plot.")
        return None, None
    df = comparison_df.copy().reset_index(drop=True)
    metrics = [m for m in metrics if m in df.columns]
    if "feature_set" not in df.columns:
        raise ValueError("comparison_df must contain a 'feature_set' column.")
    if not metrics:
        raise ValueError("None of the requested metrics are present in comparison_df.")
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.6 + 2), 4.8))
    metric_colors = make_label_color_map(metrics)
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, df[metric].astype(float).values, width=width, label=metric, color=metric_colors.get(metric))
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    labels = df["feature_set"].astype(str).tolist()
    if "n_features_used" in df.columns:
        labels = [f"{label}\n(n={int(n):,})" for label, n in zip(labels, df["n_features_used"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.05, max(0.1, np.nanmax(df[metrics].to_numpy()) * 1.12)))
    ax.set_title(title)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _maybe_savefig(save_path)
    plt.show()
    return fig, ax
