"""Classical machine-learning models and evaluation helpers."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def build_classical_models(
    random_state: int = 42,
    run_l1: bool = True,
    run_random_forest: bool = True,
) -> Dict[str, object]:
    """Build the classical models used in the project."""
    models: Dict[str, object] = {
        "Dummy_majority": DummyClassifier(strategy="most_frequent"),
        "LogReg_L2": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    if run_l1:
        models["LogReg_L1"] = OneVsRestClassifier(
            LogisticRegression(
                penalty="l1",
                C=0.5,
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
        )
    if run_random_forest:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def fit_classical_models(models: Dict[str, object], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """Fit a dictionary of sklearn-style models."""
    fitted = {}
    for name, model in models.items():
        print(f"[fit] {name}")
        fitted[name] = model.fit(X_train, y_train)
    return fitted


def _safe_predict_proba(model: object, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "macro",
) -> Dict[str, float]:
    """Evaluate a multiclass classifier with robust metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        # Weighted metrics are reported alongside macro metrics so class imbalance
        # can be discussed explicitly. Macro-F1 treats every cancer type equally;
        # weighted-F1 is influenced by larger classes.
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 2:
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            metrics["roc_auc_ovr_macro"] = np.nan
    elif y_proba is not None:
        try:
            positive_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = roc_auc_score(y_true, positive_scores)
        except Exception:
            metrics["roc_auc"] = np.nan
    return metrics


def evaluate_models(
    fitted_models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray | None]]:
    """Evaluate all fitted sklearn models.

    Returns a metrics table, predictions, and probabilities.
    """
    rows = []
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray | None] = {}
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = _safe_predict_proba(model, X_test)
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        rows.append({"model": name, **metrics})
        predictions[name] = y_pred
        probabilities[name] = y_proba
    metrics_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return metrics_df, predictions, probabilities


def classification_report_df(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    """Return sklearn classification report as a tidy DataFrame.

    ``labels`` is fixed to ``range(len(class_names))`` so rows remain aligned
    even in edge cases where a class is absent from a small test split.
    """
    labels = np.arange(len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})


def confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: str | None = None,
) -> pd.DataFrame:
    """Return a confusion matrix as a DataFrame.

    For imbalanced multiclass data, use ``normalize="true"`` for plotting.
    This normalizes each row by the number of samples in the true class, so a
    large class such as BRCA does not dominate the color scale simply because it
    has more test samples.
    """
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


def per_class_accuracy_df(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    """Return support and row-wise accuracy/recall for each class."""
    cm_count = confusion_matrix_df(y_true, y_pred, class_names, normalize=None)
    cm_norm = confusion_matrix_df(y_true, y_pred, class_names, normalize="true")
    rows = []
    for class_name in class_names:
        support = int(cm_count.loc[class_name].sum())
        correct = int(cm_count.loc[class_name, class_name])
        recall = float(cm_norm.loc[class_name, class_name]) if support > 0 else np.nan
        rows.append({"class": class_name, "support": support, "correct": correct, "class_accuracy_recall": recall})
    return pd.DataFrame(rows)


def get_best_model_name(metrics_df: pd.DataFrame, metric: str = "macro_f1") -> str:
    """Return the best model name according to a metric."""
    return metrics_df.sort_values(metric, ascending=False).iloc[0]["model"]


def classwise_accuracy_df(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    """Return per-class support, correct count, and recall-like class accuracy."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_accuracy = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)
    return pd.DataFrame(
        {
            "class": class_names,
            "support": support.astype(int),
            "correct": correct.astype(int),
            "class_accuracy": class_accuracy,
        }
    )


def top_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Return the largest off-diagonal confusion pairs.

    The ``error_rate_within_true_class`` column is row-normalized, so it is much
    more informative than raw counts when classes are imbalanced.
    """
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            rows.append(
                {
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": count,
                    "error_rate_within_true_class": float(rates[i, j]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["true_class", "predicted_class", "count", "error_rate_within_true_class"])
    return (
        pd.DataFrame(rows)
        .sort_values(["error_rate_within_true_class", "count"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
