"""Preprocessing helpers for high-dimensional gene expression matrices.

The project deliberately keeps preprocessing explicit because this is where many
mistakes in omics ML projects happen: missing values, leakage, and scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class TopVarianceGeneSelector:
    """Simple train-set-only top variance gene selector."""

    top_n: int = 3000
    selected_features_: List[str] | None = None
    variances_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "TopVarianceGeneSelector":
        variances = X.var(axis=0, skipna=True).sort_values(ascending=False)
        if self.top_n is None or self.top_n >= len(variances):
            selected = variances.index.tolist()
        else:
            selected = variances.head(self.top_n).index.tolist()
        self.variances_ = variances
        self.selected_features_ = selected
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("TopVarianceGeneSelector must be fitted before transform.")
        return X.loc[:, self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


@dataclass
class PreparedData:
    """Container for model-ready arrays and metadata."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    X_all_scaled: np.ndarray
    X_all_selected_df: pd.DataFrame
    y_all: np.ndarray
    y_all_labels: pd.Series
    feature_names: List[str]
    class_names: List[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler
    selector: TopVarianceGeneSelector
    imputer: SimpleImputer
    kept_genes_before_selection: List[str]
    cleaning_report: Dict[str, float | int] = field(default_factory=dict)


def missing_value_report(X: pd.DataFrame) -> pd.DataFrame:
    """Return a compact missing-value report for a sample x gene matrix."""
    n_values = int(X.shape[0] * X.shape[1])
    n_missing = int(X.isna().sum().sum())
    gene_missing_frac = X.isna().mean(axis=0)
    sample_missing_frac = X.isna().mean(axis=1)
    rows = [
        {"item": "n_values", "value": n_values},
        {"item": "missing_values", "value": n_missing},
        {"item": "missing_fraction", "value": n_missing / n_values if n_values else np.nan},
        {"item": "genes_with_any_missing", "value": int((gene_missing_frac > 0).sum())},
        {"item": "genes_with_>20pct_missing", "value": int((gene_missing_frac > 0.2).sum())},
        {"item": "samples_with_any_missing", "value": int((sample_missing_frac > 0).sum())},
        {"item": "max_gene_missing_fraction", "value": float(gene_missing_frac.max()) if X.shape[1] else np.nan},
        {"item": "max_sample_missing_fraction", "value": float(sample_missing_frac.max()) if X.shape[0] else np.nan},
    ]
    return pd.DataFrame(rows)



def missingness_report(X: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for ``missing_value_report``."""
    return missing_value_report(X)


def summarize_dataset(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Return a compact dataset summary table."""
    n_values = int(X.shape[0] * X.shape[1])
    n_missing = int(X.isna().sum().sum())
    rows = [
        {"item": "n_samples", "value": int(X.shape[0])},
        {"item": "n_genes", "value": int(X.shape[1])},
        {"item": "n_classes", "value": int(y.nunique())},
        {"item": "missing_values", "value": n_missing},
        {"item": "missing_fraction", "value": n_missing / n_values if n_values else np.nan},
    ]
    return pd.DataFrame(rows)


def class_count_table(y: pd.Series) -> pd.DataFrame:
    """Return class counts, percentages, and imbalance information."""
    counts = (
        y.value_counts()
        .rename_axis("cancer_type")
        .reset_index(name="n_samples")
        .sort_values("cancer_type")
        .reset_index(drop=True)
    )
    total = counts["n_samples"].sum()
    counts["sample_percent"] = (counts["n_samples"] / total * 100).round(2) if total else 0.0
    return counts


def stratified_sample_indices(
    labels: pd.Series | np.ndarray | List[str],
    max_samples: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Return row positions for a stratified subsample.

    This is useful for large t-SNE/UMAP figures: the model can train on all
    selected samples, while visualization uses a smaller but class-balanced
    subset so that large classes do not visually dominate the plot.
    """
    labels_series = pd.Series(labels).reset_index(drop=True)
    n_total = len(labels_series)
    if max_samples is None or max_samples >= n_total:
        return np.arange(n_total)

    rng = np.random.default_rng(random_state)
    classes = labels_series.dropna().unique().tolist()
    n_classes = max(len(classes), 1)
    per_class = max(1, max_samples // n_classes)

    chosen: List[np.ndarray] = []
    remaining: List[np.ndarray] = []
    for cls in classes:
        idx = labels_series.index[labels_series == cls].to_numpy()
        rng.shuffle(idx)
        take = min(len(idx), per_class)
        chosen.append(idx[:take])
        if take < len(idx):
            remaining.append(idx[take:])

    chosen_idx = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    # If some classes had fewer samples, fill the remaining quota from the other classes.
    n_left = max_samples - len(chosen_idx)
    if n_left > 0 and remaining:
        pool = np.concatenate(remaining)
        rng.shuffle(pool)
        chosen_idx = np.concatenate([chosen_idx, pool[:n_left]])

    return np.sort(chosen_idx.astype(int))


def clean_expression_matrix(
    X: pd.DataFrame,
    max_missing_fraction: float = 0.2,
    min_variance: float = 0.0,
) -> pd.DataFrame:
    """Clean a full expression matrix for EDA-only use.

    This function is intentionally simple and can use the whole matrix because it
    is meant for descriptive EDA. For model training, use ``prepare_model_data``;
    that function fits imputation, feature selection, and scaling only on the
    train set to avoid leakage.
    """
    X = X.copy().replace([np.inf, -np.inf], np.nan)
    min_non_missing = int(np.ceil((1.0 - max_missing_fraction) * len(X)))
    X = X.dropna(axis=1, thresh=min_non_missing)

    if X.isna().any().any():
        medians = X.median(axis=0, numeric_only=True)
        X = X.fillna(medians)

    variances = X.var(axis=0)
    X = X.loc[:, variances > min_variance]
    return X


def prepare_eda_matrix(
    X: pd.DataFrame,
    top_n_genes: int = 3000,
    max_missing_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, List[str], pd.DataFrame]:
    """Prepare a scaled matrix for PCA/t-SNE/UMAP during EDA.

    EDA embeddings cannot handle NaN values. We therefore drop genes with too
    much missingness, median-impute the remaining small amount of missingness,
    select high-variance genes, and scale them.

    This is for visualization only. Model training uses train-set-only fitting.
    """
    del random_state  # kept for a stable public signature
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan)
    min_non_missing = int(np.ceil((1.0 - max_missing_fraction) * len(X_clean)))
    X_clean = X_clean.dropna(axis=1, thresh=min_non_missing)

    medians = X_clean.median(axis=0, numeric_only=True)
    X_clean = X_clean.fillna(medians)

    variances = X_clean.var(axis=0).sort_values(ascending=False)
    if top_n_genes is not None:
        selected_genes = variances.head(min(top_n_genes, len(variances))).index.tolist()
    else:
        selected_genes = variances.index.tolist()

    X_selected = X_clean.loc[:, selected_genes]
    X_scaled = StandardScaler().fit_transform(X_selected).astype("float32")

    report = pd.DataFrame(
        [
            {"item": "eda_input_genes", "value": int(X.shape[1])},
            {"item": "eda_genes_after_missing_filter", "value": int(X_clean.shape[1])},
            {"item": "eda_selected_top_variable_genes", "value": int(len(selected_genes))},
            {"item": "eda_remaining_missing_values", "value": int(np.isnan(X_scaled).sum())},
        ]
    )
    return X_scaled, selected_genes, report


def _filter_genes_from_train(
    X_train: pd.DataFrame,
    max_missing_fraction: float,
    min_variance: float,
) -> List[str]:
    """Select genes using train-set-only missingness and variance rules."""
    missing_fraction = X_train.isna().mean(axis=0)
    keep = missing_fraction <= max_missing_fraction
    candidate_genes = missing_fraction.index[keep].tolist()

    # Estimate variance after median imputation, using train data only.
    X_candidate = X_train.loc[:, candidate_genes]
    medians = X_candidate.median(axis=0, numeric_only=True)
    X_imputed = X_candidate.fillna(medians)
    variances = X_imputed.var(axis=0)
    candidate_genes = variances.index[variances > min_variance].tolist()
    return candidate_genes


def prepare_model_data(
    X: pd.DataFrame,
    y: pd.Series,
    top_n_genes: int = 3000,
    test_size: float = 0.2,
    random_state: int = 42,
    max_missing_fraction: float = 0.2,
    min_variance: float = 0.0,
) -> PreparedData:
    """Split, impute, select genes, scale, and encode labels.

    Leakage-safe order:
    1. Split train/test first.
    2. Decide which genes to keep using the train set only.
    3. Fit median imputer on the train set only.
    4. Select top-variance genes using the train set only.
    5. Fit scaler on the train set only.
    """
    X = X.copy().replace([np.inf, -np.inf], np.nan)
    y = y.loc[X.index]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    class_names = label_encoder.classes_.tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    kept_genes = _filter_genes_from_train(
        X_train_raw,
        max_missing_fraction=max_missing_fraction,
        min_variance=min_variance,
    )
    X_train_keep = X_train_raw.loc[:, kept_genes]
    X_test_keep = X_test_raw.loc[:, kept_genes]
    X_all_keep = X.loc[:, kept_genes]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_keep),
        index=X_train_keep.index,
        columns=X_train_keep.columns,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_keep),
        index=X_test_keep.index,
        columns=X_test_keep.columns,
    )
    X_all_imp = pd.DataFrame(
        imputer.transform(X_all_keep),
        index=X_all_keep.index,
        columns=X_all_keep.columns,
    )

    selector = TopVarianceGeneSelector(top_n=top_n_genes)
    X_train_sel = selector.fit_transform(X_train_imp)
    X_test_sel = selector.transform(X_test_imp)
    X_all_sel = selector.transform(X_all_imp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel).astype("float32")
    X_test_scaled = scaler.transform(X_test_sel).astype("float32")
    X_all_scaled = scaler.transform(X_all_sel).astype("float32")

    X_train_df = pd.DataFrame(X_train_scaled, index=X_train_sel.index, columns=X_train_sel.columns)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test_sel.index, columns=X_test_sel.columns)
    X_all_selected_df = pd.DataFrame(X_all_scaled, index=X_all_sel.index, columns=X_all_sel.columns)

    y_all = label_encoder.transform(y.astype(str))
    original_values = int(X.shape[0] * X.shape[1])
    original_missing = int(X.isna().sum().sum())
    cleaning_report: Dict[str, float | int] = {
        "original_genes": int(X.shape[1]),
        "original_missing_values": original_missing,
        "original_missing_fraction": original_missing / original_values if original_values else np.nan,
        "genes_after_missing_variance_filter": int(len(kept_genes)),
        "selected_top_variable_genes": int(len(selector.selected_features_ or [])),
        "train_remaining_missing_values": int(np.isnan(X_train_scaled).sum()),
        "test_remaining_missing_values": int(np.isnan(X_test_scaled).sum()),
    }

    return PreparedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        X_all_scaled=X_all_scaled,
        X_all_selected_df=X_all_selected_df,
        y_all=y_all,
        y_all_labels=y,
        feature_names=X_train_sel.columns.tolist(),
        class_names=class_names,
        label_encoder=label_encoder,
        scaler=scaler,
        selector=selector,
        imputer=imputer,
        kept_genes_before_selection=kept_genes,
        cleaning_report=cleaning_report,
    )
