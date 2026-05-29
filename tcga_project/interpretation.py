"""Model interpretation helpers: coefficients and feature importances."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _coef_matrix(model) -> np.ndarray:
    """Return coefficient matrix from LogisticRegression or OneVsRestClassifier."""
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_)
    if hasattr(model, "estimators_"):
        coefs = []
        for est in model.estimators_:
            if not hasattr(est, "coef_"):
                raise TypeError("One estimator does not have coef_.")
            coefs.append(np.asarray(est.coef_).ravel())
        return np.vstack(coefs)
    raise TypeError("The provided model does not have coefficients.")


def logistic_top_genes(model, feature_names: List[str], class_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """Extract top positive coefficients per class from a fitted LogisticRegression model."""
    coefs = _coef_matrix(model)
    rows = []

    if coefs.shape[0] == 1 and len(class_names) == 2:
        # Binary sklearn convention: one coefficient vector for class 1 vs class 0.
        coef = coefs[0]
        for sign, label in [(1, class_names[1]), (-1, class_names[0])]:
            signed = coef * sign
            order = np.argsort(signed)[::-1][:top_n]
            for rank, idx in enumerate(order, start=1):
                rows.append(
                    {
                        "class": label,
                        "rank": rank,
                        "gene": feature_names[idx],
                        "coef": float(coef[idx]),
                        "class_direction_score": float(signed[idx]),
                        "abs_coef": float(abs(coef[idx])),
                    }
                )
    else:
        for class_idx, class_name in enumerate(class_names):
            coef = coefs[class_idx]
            order = np.argsort(coef)[::-1][:top_n]
            for rank, idx in enumerate(order, start=1):
                rows.append(
                    {
                        "class": class_name,
                        "rank": rank,
                        "gene": feature_names[idx],
                        "coef": float(coef[idx]),
                        "class_direction_score": float(coef[idx]),
                        "abs_coef": float(abs(coef[idx])),
                    }
                )
    return pd.DataFrame(rows)


def logistic_nonzero_genes(model, feature_names: List[str], threshold: float = 1e-8) -> pd.DataFrame:
    """Summarize genes with nonzero coefficients in an L1 logistic model."""
    coefs = _coef_matrix(model)
    abs_max = np.max(np.abs(coefs), axis=0)
    df = pd.DataFrame({"gene": feature_names, "max_abs_coef": abs_max})
    return df[df["max_abs_coef"] > threshold].sort_values("max_abs_coef", ascending=False).reset_index(drop=True)


def random_forest_top_genes(model, feature_names: List[str], top_n: int = 30) -> pd.DataFrame:
    """Extract top feature importances from a fitted RandomForestClassifier."""
    if not hasattr(model, "feature_importances_"):
        raise TypeError("The provided model does not have feature_importances_.")
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1][:top_n]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "gene": [feature_names[i] for i in order],
            "importance": importances[order],
        }
    )


def overlapping_top_genes(*tables: pd.DataFrame, gene_col: str = "gene") -> pd.DataFrame:
    """Find genes that appear in multiple top-gene tables."""
    counts: Dict[str, int] = {}
    for table in tables:
        if table is None or table.empty or gene_col not in table.columns:
            continue
        for gene in table[gene_col].dropna().astype(str).unique():
            counts[gene] = counts.get(gene, 0) + 1
    rows = [{"gene": gene, "n_tables": n} for gene, n in counts.items()]
    return pd.DataFrame(rows).sort_values(["n_tables", "gene"], ascending=[False, True]).reset_index(drop=True)
