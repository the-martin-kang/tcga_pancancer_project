"""Reusable experiment runners for the TCGA Pan-Cancer project."""
from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .models import evaluate_predictions
from .preprocessing import prepare_model_data


def _get_tqdm(show_progress: bool):
    """Import tqdm lazily so the project still runs if tqdm is unavailable."""
    if not show_progress:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm
    except Exception:
        return None


def _fit_with_alive_progress(
    model,
    X_train,
    y_train,
    *,
    desc: str,
    show_progress: bool = True,
    update_interval: float = 2.0,
):
    """Fit a scikit-learn model while showing an alive progress indicator.

    scikit-learn's LogisticRegression does not expose per-iteration callbacks,
    so an exact percent-complete tqdm bar is not possible. Instead, fitting is
    executed in a background thread and tqdm is updated every few seconds. This
    makes long all-gene experiments visibly active in Jupyter notebooks.
    """
    tqdm = _get_tqdm(show_progress)
    if tqdm is None:
        return model.fit(X_train, y_train)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.fit, X_train, y_train)
        with tqdm(total=None, desc=desc, unit="tick", leave=True) as pbar:
            while not future.done():
                time.sleep(update_interval)
                pbar.update(1)
            future.result()  # re-raise exceptions from fit()
    return model


def run_feature_set_size_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_missing_fraction: float = 0.2,
    min_variance: float = 0.0,
    show_progress: bool = True,
    fit_progress_interval: float = 2.0,
    solver_verbose: int = 0,
) -> pd.DataFrame:
    """Compare Top-N genes with all usable genes using L2 Logistic Regression.

    ``top_n_genes=None`` means all genes retained after train-set-only
    missingness and variance filtering. The SAGA solver is used because it is
    usually more practical for the all-gene high-dimensional setting.

    Progress behavior:
    - tqdm over feature-set specifications shows the overall experiment status.
    - A second alive tqdm indicator updates while the long LogisticRegression
      fit() call is running. It is elapsed-time based, not percent-complete.
    """
    rows = []
    tqdm = _get_tqdm(show_progress)
    iterator = feature_sets
    if tqdm is not None:
        iterator = tqdm(feature_sets, desc="Feature-set comparison", unit="set")

    for spec in iterator:
        if isinstance(spec, dict):
            display_name = spec.get("name", "feature_set")
            top_n = spec.get("top_n_genes", None)
        else:
            display_name, top_n = spec

        msg = f"[feature-set comparison] {display_name}"
        if tqdm is not None:
            tqdm.write("\n" + msg)
        else:
            print("\n" + msg)

        start = time.perf_counter()
        stage_bar = None
        prepared = None
        model = None
        y_pred = None
        y_proba = None

        try:
            if tqdm is not None:
                stage_bar = tqdm(total=3, desc=f"{display_name}: stages", unit="stage", leave=False)
                stage_bar.set_postfix_str("preprocess")

            prepared = prepare_model_data(
                X,
                y,
                top_n_genes=top_n,
                test_size=test_size,
                random_state=random_state,
                max_missing_fraction=max_missing_fraction,
                min_variance=min_variance,
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix_str(f"fit {len(prepared.feature_names):,} genes")

            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="saga",
                max_iter=700,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbose=solver_verbose,
            )

            _fit_with_alive_progress(
                model,
                prepared.X_train,
                prepared.y_train,
                desc=f"Fitting {display_name}",
                show_progress=show_progress,
                update_interval=fit_progress_interval,
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix_str("evaluate")

            y_pred = model.predict(prepared.X_test)
            y_proba = model.predict_proba(prepared.X_test) if hasattr(model, "predict_proba") else None
            metrics = evaluate_predictions(prepared.y_test, y_pred, y_proba)
            if stage_bar is not None:
                stage_bar.update(1)

            rows.append({
                "feature_set": display_name,
                "model": "LogReg_L2_saga",
                "requested_top_n_genes": "all" if top_n is None else int(top_n),
                "n_original_genes": int(X.shape[1]),
                "n_genes_after_missing_variance_filter": int(len(prepared.kept_genes_before_selection)),
                "n_features_used": int(len(prepared.feature_names)),
                "missing_fraction_original": float(prepared.cleaning_report.get("original_missing_fraction", np.nan)),
                "runtime_seconds": float(time.perf_counter() - start),
                **metrics,
            })

            done_msg = (
                f"done: {display_name} | features={len(prepared.feature_names):,} | "
                f"macro_f1={metrics.get('macro_f1', np.nan):.4f} | "
                f"runtime={time.perf_counter() - start:.1f}s"
            )
            if tqdm is not None:
                tqdm.write(done_msg)
            else:
                print(done_msg)
        finally:
            if stage_bar is not None:
                stage_bar.close()
            del prepared, model, y_pred, y_proba
            gc.collect()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
