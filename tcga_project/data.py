"""Data download, parsing, and loading for TCGA Pan-Cancer RNA-seq data."""
from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CDR_FILENAME, GDC_CDR_URL, GDC_RNA_URL, RNA_FILENAME, ProjectConfig, DEFAULT_CANCER_TYPES_20, CANCER_FULL_NAMES


def download_file(url: str, destination: str | Path, overwrite: bool = False, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file from URL to destination.

    The RNA expression file is large. The function streams the download and skips
    existing files by default.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[skip] {destination.name} already exists: {destination}")
        return destination

    print(f"[download] {url}\n        -> {destination}")
    tmp = destination.with_suffix(destination.suffix + ".part")

    try:
        import requests
        from tqdm.auto import tqdm

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as exc:
        print(f"[warning] requests/tqdm download failed ({exc}). Falling back to urllib.")
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f)

    tmp.replace(destination)
    return destination


def ensure_gdc_files(config: ProjectConfig, overwrite: bool = False) -> Tuple[Path, Path]:
    """Ensure that the official GDC RNA and clinical files exist locally."""
    config.ensure_dirs()
    rna_path = download_file(GDC_RNA_URL, config.raw_dir / RNA_FILENAME, overwrite=overwrite)
    cdr_path = download_file(GDC_CDR_URL, config.raw_dir / CDR_FILENAME, overwrite=overwrite)
    return rna_path, cdr_path


def read_table_header(path: str | Path, sep: str = "\t") -> List[str]:
    """Read only the header line of a delimited text file, including .gz files."""
    path = Path(path)
    compression = "gzip" if path.suffix == ".gz" else None
    header = pd.read_csv(path, sep=sep, nrows=0, compression=compression).columns.tolist()
    return header


def tcga_patient_barcode(sample_id: str) -> str:
    """Return the TCGA patient barcode, usually the first 12 characters.

    Example: TCGA-AB-1234-01A-... -> TCGA-AB-1234
    """
    s = str(sample_id).strip().replace("_", "-")
    return s[:12].upper()


def tcga_sample_barcode(sample_id: str) -> str:
    """Return the normalized TCGA sample barcode, usually the first 15 characters."""
    s = str(sample_id).strip().replace("_", "-")
    return s[:15].upper()


def tcga_sample_type_code(sample_id: str) -> str | None:
    """Extract TCGA sample type code from a sample barcode.

    Primary tumor samples have code '01'. Solid tissue normal samples have code '11'.
    """
    s = str(sample_id).strip().replace("_", "-")
    parts = s.split("-")
    if len(parts) >= 4 and len(parts[3]) >= 2:
        return parts[3][:2]
    if len(s) >= 15:
        return s[13:15]
    return None


def load_clinical_cdr(path: str | Path) -> pd.DataFrame:
    """Load the TCGA-CDR clinical file and standardize key column names.

    Expected important columns in the original file include:
    - bcr_patient_barcode
    - type
    - OS, OS.time, etc. (not required for the main classification task)
    """
    path = Path(path)
    clinical = pd.read_excel(path)
    clinical.columns = [str(c).strip() for c in clinical.columns]

    # Find patient barcode column.
    patient_candidates = ["bcr_patient_barcode", "patient", "submitter_id", "case_submitter_id"]
    patient_col = next((c for c in patient_candidates if c in clinical.columns), None)
    if patient_col is None:
        # Fallback: look for a column that contains TCGA-like barcodes.
        for c in clinical.columns:
            sample_values = clinical[c].dropna().astype(str).head(20)
            if sample_values.str.startswith("TCGA-").any():
                patient_col = c
                break
    if patient_col is None:
        raise ValueError("Could not find a TCGA patient barcode column in the clinical CDR file.")

    type_candidates = ["type", "cancer_type", "project_id", "cohort", "study"]
    type_col = next((c for c in type_candidates if c in clinical.columns), None)
    if type_col is None:
        raise ValueError("Could not find a cancer type column such as 'type' in the clinical CDR file.")

    clinical = clinical.copy()
    clinical["patient_barcode"] = clinical[patient_col].map(tcga_patient_barcode)
    clinical["cancer_type"] = clinical[type_col].astype(str).str.upper().str.replace("TCGA-", "", regex=False)
    clinical = clinical.drop_duplicates(subset=["patient_barcode"], keep="first")
    return clinical


def make_sample_table_from_expression_header(
    expression_path: str | Path,
    clinical: pd.DataFrame,
    cancer_types: Sequence[str] | None = None,
    n_top_cancer_types: int | None = None,
    primary_tumor_only: bool = True,
    max_samples_per_class: int | None = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a sample table by matching RNA-seq columns to TCGA-CDR labels.

    Parameters
    ----------
    cancer_types
        Explicit cancer-type abbreviations to keep. If None, all matched cancer
        types are considered first.
    n_top_cancer_types
        If provided, keep the top N cancer types by sample count after primary
        tumor filtering and one-sample-per-patient deduplication. This is useful
        for a larger 20-class project without hand-picking cancer types.
    """
    header = read_table_header(expression_path, sep="\t")
    if len(header) < 2:
        raise ValueError("The expression file header does not look like a gene x sample matrix.")

    gene_col = header[0]
    sample_cols = header[1:]
    patient_to_type = dict(zip(clinical["patient_barcode"], clinical["cancer_type"]))
    wanted = None if cancer_types is None else {c.upper() for c in cancer_types}

    records = []
    for col in sample_cols:
        patient = tcga_patient_barcode(col)
        cancer_type = patient_to_type.get(patient)
        if cancer_type is None:
            continue
        if wanted is not None and cancer_type not in wanted:
            continue
        sample_type = tcga_sample_type_code(col)
        if primary_tumor_only and sample_type != "01":
            continue
        records.append(
            {
                "sample_col": col,
                "sample_barcode": tcga_sample_barcode(col),
                "patient_barcode": patient,
                "sample_type_code": sample_type,
                "cancer_type": cancer_type,
            }
        )

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError(
            "No matched TCGA RNA-seq samples were found. Check that the expression file and clinical CDR file match."
        )

    # Keep one primary tumor sample per patient to avoid duplicate patient leakage.
    table = table.drop_duplicates(subset=["patient_barcode"], keep="first")

    # Optional automatic top-N class selection. This intentionally happens before
    # max_samples_per_class capping so that the top-N selection reflects the raw
    # available sample counts.
    if n_top_cancer_types is not None:
        raw_counts = table["cancer_type"].value_counts()
        keep_types = raw_counts.head(n_top_cancer_types).index.tolist()
        table = table[table["cancer_type"].isin(keep_types)].copy()
        table.attrs["auto_selected_cancer_types"] = tuple(keep_types)

    # Runtime control by downsampling very large classes. This does not fully
    # balance the classes; it only caps very large classes. Class imbalance is
    # intentionally preserved for realistic evaluation.
    if max_samples_per_class is not None:
        sampled_groups = []
        for _, group in table.groupby("cancer_type"):
            sampled_groups.append(group.sample(n=min(len(group), max_samples_per_class), random_state=random_state))
        table = pd.concat(sampled_groups, axis=0).reset_index(drop=True)

    # Stable ordering is useful for reproducibility.
    table = table.sort_values(["cancer_type", "sample_col"]).reset_index(drop=True)
    table.attrs["gene_column"] = gene_col
    table.attrs["selected_cancer_types"] = tuple(sorted(table["cancer_type"].unique()))
    return table


def clean_gene_name(gene_id: str) -> str:
    """Clean a gene identifier into a readable gene symbol when possible."""
    s = str(gene_id).strip()
    # Common PanCanAtlas format: SYMBOL|EntrezID
    if "|" in s:
        s = s.split("|")[0]
    # Remove Ensembl version suffix if present.
    s = re.sub(r"\.\d+$", "", s)
    return s


def load_expression_subset(
    expression_path: str | Path,
    sample_table: pd.DataFrame,
    gene_column: str | None = None,
    dtype: str = "float32",
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load only selected sample columns from the large PanCanAtlas expression matrix.

    Returns a sample x gene matrix with sample column names as the index.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if use_cache and cache_path.exists():
            print(f"[cache] Loading processed expression matrix: {cache_path}")
            return pd.read_pickle(cache_path)

    expression_path = Path(expression_path)
    header = read_table_header(expression_path, sep="\t")
    gene_column = gene_column or sample_table.attrs.get("gene_column") or header[0]
    selected_cols = sample_table["sample_col"].tolist()
    missing_cols = sorted(set(selected_cols) - set(header))
    if missing_cols:
        raise ValueError(f"{len(missing_cols)} requested sample columns are missing from the expression file.")

    usecols = [gene_column] + selected_cols
    compression = "gzip" if expression_path.suffix == ".gz" else None
    dtype_map = {col: dtype for col in selected_cols}

    print(f"[load] Reading {len(selected_cols)} selected samples from RNA matrix...")
    raw = pd.read_csv(expression_path, sep="\t", usecols=usecols, compression=compression, dtype=dtype_map)
    raw = raw.rename(columns={gene_column: "gene"})
    raw["gene"] = raw["gene"].map(clean_gene_name)
    raw = raw.dropna(subset=["gene"])
    raw = raw[raw["gene"] != "?"]
    raw = raw[raw["gene"].astype(str).str.len() > 0]

    # Average duplicate gene symbols if any exist.
    expr_gene_by_sample = raw.groupby("gene", sort=False)[selected_cols].mean(numeric_only=True)
    X = expr_gene_by_sample.T
    X.index.name = "sample_col"
    X = X.astype(dtype)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        X.to_pickle(cache_path)
        print(f"[cache] Saved processed expression matrix: {cache_path}")
    return X


def prepare_tcga_expression_dataset(
    config: ProjectConfig,
    overwrite_download: bool = False,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """End-to-end data loading helper for the notebook.

    Returns
    -------
    X : pd.DataFrame
        sample x gene expression matrix.
    y : pd.Series
        cancer type label for each sample.
    sample_table : pd.DataFrame
        sample metadata used for loading and labels.
    """
    config.ensure_dirs()
    rna_path, cdr_path = ensure_gdc_files(config, overwrite=overwrite_download)
    clinical = load_clinical_cdr(cdr_path)
    sample_table = make_sample_table_from_expression_header(
        rna_path,
        clinical,
        cancer_types=config.cancer_types,
        n_top_cancer_types=config.n_top_cancer_types,
        primary_tumor_only=True,
        max_samples_per_class=config.max_samples_per_class,
        random_state=config.random_state,
    )

    selected_types = tuple(sorted(sample_table["cancer_type"].unique()))
    cancer_tag = "-".join(selected_types)
    sample_tag = f"max{config.max_samples_per_class}" if config.max_samples_per_class else "all"
    top_tag = f"top{config.n_top_cancer_types}" if config.n_top_cancer_types else "fixed"
    cache_key = "|".join([cancer_tag, sample_tag, top_tag])
    cache_name = f"expression_subset_{hashlib.md5(cache_key.encode()).hexdigest()[:8]}.pkl"
    cache_path = config.processed_dir / cache_name
    X = load_expression_subset(
        rna_path,
        sample_table,
        gene_column=sample_table.attrs.get("gene_column"),
        cache_path=cache_path,
        use_cache=use_cache,
    )
    y = sample_table.set_index("sample_col").loc[X.index, "cancer_type"]
    y.name = "cancer_type"
    return X, y, sample_table


def make_demo_gene_expression(
    n_classes: int = 5,
    samples_per_class: int = 80,
    n_genes: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Generate a small synthetic gene expression dataset for testing the pipeline without downloading TCGA."""
    rng = np.random.default_rng(random_state)
    preferred = list(DEFAULT_CANCER_TYPES_20)
    extra = [c for c in CANCER_FULL_NAMES.keys() if c not in preferred]
    cancer_pool = preferred + extra
    if n_classes > len(cancer_pool):
        cancer_pool += [f"CANCER_{i:02d}" for i in range(len(cancer_pool) + 1, n_classes + 1)]
    cancer_types = cancer_pool[:n_classes]

    X_blocks = []
    labels = []
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    for i, cancer in enumerate(cancer_types):
        base = rng.normal(0, 1, size=(samples_per_class, n_genes))
        # Add class-specific expression shift in a small gene block.
        start = (i * 30) % max(n_genes, 1)
        end = min(start + 50, n_genes)
        if start < end:
            base[:, start:end] += rng.normal(2.5, 0.3)
        X_blocks.append(base)
        labels += [cancer] * samples_per_class
    X_arr = np.vstack(X_blocks).astype("float32")
    sample_ids = [f"DEMO-{i:04d}" for i in range(X_arr.shape[0])]
    X = pd.DataFrame(X_arr, index=sample_ids, columns=genes)
    y = pd.Series(labels, index=sample_ids, name="cancer_type")
    sample_table = pd.DataFrame({"sample_col": sample_ids, "cancer_type": labels})
    return X, y, sample_table
