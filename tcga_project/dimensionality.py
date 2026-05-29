"""Dimensionality reduction and clustering utilities."""
from __future__ import annotations

import inspect
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def _pre_pca_if_needed(X: np.ndarray, n_components: int = 50, random_state: int = 42) -> np.ndarray:
    """Reduce very high-dimensional data before t-SNE/UMAP for speed."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    max_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if max_components >= 2 and X.shape[1] > max_components:
        return PCA(n_components=max_components, random_state=random_state).fit_transform(X)
    return X


def compute_pca(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> Tuple[np.ndarray, PCA]:
    """Compute PCA embedding."""
    pca = PCA(n_components=n_components, random_state=random_state)
    emb = pca.fit_transform(X)
    return emb, pca


def stratified_sample_indices(
    labels: Iterable,
    max_samples: int | None = None,
    max_per_class: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Return positional indices from a stratified sample.

    t-SNE/UMAP can be slow and visually cluttered on thousands of TCGA samples.
    This helper keeps class proportions approximately intact while limiting
    runtime. If both ``max_samples`` and ``max_per_class`` are None, all indices
    are returned.
    """
    y = pd.Series(list(labels)).reset_index(drop=True)
    n = len(y)
    if max_samples is None and max_per_class is None:
        return np.arange(n)
    if max_samples is not None and n <= max_samples and max_per_class is None:
        return np.arange(n)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or None.")
    if max_per_class is not None and max_per_class <= 0:
        raise ValueError("max_per_class must be positive or None.")

    rng = np.random.default_rng(random_state)
    selected: list[int] = []

    if max_per_class is not None:
        for _, idx in y.groupby(y).groups.items():
            idx_arr = np.array(list(idx), dtype=int)
            take = min(max_per_class, len(idx_arr))
            selected.extend(rng.choice(idx_arr, size=take, replace=False).tolist())
    else:
        counts = y.value_counts()
        alloc = np.floor(counts / counts.sum() * int(max_samples)).astype(int)
        alloc[alloc < 1] = 1
        alloc = np.minimum(alloc, counts)

        # Adjust allocation to be close to max_samples while respecting class sizes.
        while alloc.sum() < max_samples:
            room = counts - alloc
            if room.max() <= 0:
                break
            cls = room.idxmax()
            alloc.loc[cls] += 1
        while alloc.sum() > max_samples:
            candidates = alloc[alloc > 1]
            if candidates.empty:
                break
            cls = candidates.idxmax()
            alloc.loc[cls] -= 1

        for cls, take in alloc.items():
            idx_arr = np.array(y.index[y == cls], dtype=int)
            selected.extend(rng.choice(idx_arr, size=int(take), replace=False).tolist())

    return np.array(sorted(selected), dtype=int)


def stratified_subsample_indices(labels: Iterable, max_samples: int | None = None, random_state: int = 42) -> np.ndarray:
    """Backward-compatible alias for ``stratified_sample_indices``."""
    return stratified_sample_indices(labels, max_samples=max_samples, random_state=random_state)


def compute_tsne(
    X: np.ndarray,
    perplexity: int = 30,
    random_state: int = 42,
    pre_pca_components: int = 50,
) -> np.ndarray:
    """Compute a 2D t-SNE embedding.

    The function first reduces the data to at most 50 PCA components for speed,
    which is a common practical workflow for high-dimensional expression data.
    """
    X_reduced = _pre_pca_if_needed(X, n_components=pre_pca_components, random_state=random_state)
    perplexity = min(perplexity, max(5, (X_reduced.shape[0] - 1) // 3))

    kwargs = dict(n_components=2, perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto")
    sig = inspect.signature(TSNE)
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    return TSNE(**kwargs).fit_transform(X_reduced)


def compute_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    pre_pca_components: int = 50,
) -> np.ndarray:
    """Compute a 2D UMAP embedding if umap-learn is installed."""
    try:
        import umap
    except Exception as exc:
        raise ImportError("umap-learn is not installed. Install with: pip install umap-learn") from exc

    X_reduced = _pre_pca_if_needed(X, n_components=pre_pca_components, random_state=random_state)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(X_reduced)


def compute_embedding(
    X: np.ndarray,
    method: str = "tsne",
    random_state: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, str]:
    """Compute PCA, t-SNE, or UMAP embedding with a safe UMAP fallback to t-SNE."""
    method = method.lower()
    if method == "pca":
        emb, _ = compute_pca(X, n_components=2, random_state=random_state)
        return emb, "PCA"
    if method == "umap":
        try:
            return compute_umap(X, random_state=random_state, **kwargs), "UMAP"
        except ImportError as exc:
            print(f"[warning] {exc}. Falling back to t-SNE.")
            return compute_tsne(X, random_state=random_state), "t-SNE"
    if method in {"tsne", "t-sne"}:
        return compute_tsne(X, random_state=random_state, **kwargs), "t-SNE"
    raise ValueError("method must be one of: 'pca', 'tsne', 'umap'.")


def embedding_dataframe(embedding: np.ndarray, labels: Iterable, method_name: str) -> pd.DataFrame:
    """Build a tidy DataFrame for plotting a 2D embedding."""
    return pd.DataFrame(
        {
            "Dim1": embedding[:, 0],
            "Dim2": embedding[:, 1],
            "label": list(labels),
            "method": method_name,
        }
    )


def evaluate_kmeans_space(
    X_space: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    name: str = "space",
) -> Dict[str, float | str]:
    """Run KMeans in a given representation and compare clusters with true labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    cluster = kmeans.fit_predict(X_space)
    row: Dict[str, float | str] = {
        "space": name,
        "ARI": adjusted_rand_score(y_true, cluster),
        "NMI": normalized_mutual_info_score(y_true, cluster),
    }
    try:
        row["silhouette"] = silhouette_score(X_space, cluster)
    except Exception:
        row["silhouette"] = np.nan
    return row


def compare_kmeans_spaces(
    spaces: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare KMeans clustering performance across multiple feature spaces."""
    rows = [evaluate_kmeans_space(X, y_true, n_clusters, random_state, name) for name, X in spaces.items()]
    return pd.DataFrame(rows)
