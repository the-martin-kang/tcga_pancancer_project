"""Configuration values for the TCGA Pan-Cancer classroom project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

# Official GDC PanCanAtlas supplemental file links.
# GDC page: https://gdc.cancer.gov/about-data/publications/pancanatlas
GDC_RNA_URL = "https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611"
GDC_CDR_URL = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"

RNA_FILENAME = "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
CDR_FILENAME = "TCGA-CDR-SupplementalTableS1.xlsx"

DEFAULT_CANCER_TYPES_5: Tuple[str, ...] = ("BRCA", "KIRC", "COAD", "LUAD", "PRAD")

DEFAULT_CANCER_TYPES_10: Tuple[str, ...] = (
    "BRCA", "KIRC", "COAD", "LUAD", "PRAD",
    "STAD", "LIHC", "THCA", "HNSC", "UCEC",
)

# Larger but still manageable setting for a 48GB-RAM local laptop.
# It intentionally includes similar pairs such as LUAD/LUSC and KIRC/KIRP so
# that error patterns are more interesting than in a small 5-class task.
DEFAULT_CANCER_TYPES_20: Tuple[str, ...] = (
    "BRCA", "KIRC", "COAD", "LUAD", "PRAD",
    "STAD", "LIHC", "THCA", "HNSC", "UCEC",
    "BLCA", "LUSC", "KIRP", "CESC", "ESCA",
    "PAAD", "SARC", "SKCM", "OV", "LGG",
)

DEFAULT_CANCER_TYPES_33: Tuple[str, ...] = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)

CANCER_TYPE_SETS: Dict[str, Tuple[str, ...]] = {
    "5": DEFAULT_CANCER_TYPES_5,
    "10": DEFAULT_CANCER_TYPES_10,
    "20": DEFAULT_CANCER_TYPES_20,
    "33": DEFAULT_CANCER_TYPES_33,
    "all": DEFAULT_CANCER_TYPES_33,
}

CANCER_FULL_NAMES: Dict[str, str] = {
    "ACC": "Adrenocortical carcinoma",
    "BLCA": "Bladder urothelial carcinoma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon adenocarcinoma",
    "DLBC": "Lymphoid neoplasm diffuse large B-cell lymphoma",
    "ESCA": "Esophageal carcinoma",
    "GBM": "Glioblastoma multiforme",
    "HNSC": "Head and neck squamous cell carcinoma",
    "KICH": "Kidney chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma",
    "KIRP": "Kidney renal papillary cell carcinoma",
    "LAML": "Acute myeloid leukemia",
    "LGG": "Brain lower grade glioma",
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "READ": "Rectum adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin cutaneous melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular germ cell tumors",
    "THCA": "Thyroid carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine corpus endometrial carcinoma",
    "UCS": "Uterine carcinosarcoma",
    "UVM": "Uveal melanoma",
}


def resolve_cancer_types(name: str | Sequence[str]) -> Tuple[str, ...]:
    """Resolve a preset name such as "5", "10", "20", or "33".

    A custom sequence of TCGA abbreviations can also be passed.
    """
    if isinstance(name, str):
        key = name.strip().lower()
        if key not in CANCER_TYPE_SETS:
            valid = ", ".join(sorted(CANCER_TYPE_SETS))
            raise ValueError(f"Unknown cancer set {name!r}. Choose one of {valid}, or pass a custom sequence.")
        return CANCER_TYPE_SETS[key]
    return tuple(str(x).upper() for x in name)


@dataclass
class ProjectConfig:
    """Central configuration for the notebook pipeline.

    Defaults are set for a 48GB-RAM local laptop. For Colab or a smaller laptop,
    reduce ``max_samples_per_class`` and ``top_n_genes`` in the notebook settings cell.
    """

    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    raw_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "processed")
    outputs_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs")
    figure_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs" / "figures")
    table_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs" / "tables")

    # If cancer_types is None and n_top_cancer_types is set, data.py will choose
    # the top N cancer types by available primary-tumor sample count.
    cancer_types: Tuple[str, ...] | None = DEFAULT_CANCER_TYPES_20
    n_top_cancer_types: int | None = None
    max_samples_per_class: int | None = 1000
    random_state: int = 42

    # Modeling scale controls
    test_size: float = 0.2
    top_n_genes: int | None = 2000
    max_missing_fraction: float = 0.2
    max_samples_for_embedding: int | None = 3000

    # Runtime switches
    demo_mode: bool = False
    run_l1_logreg: bool = True
    run_random_forest: bool = True
    run_mlp: bool = True
    embedding_method: str = "umap"  # UMAP preferred for larger runs; falls back to t-SNE if unavailable.

    # MLP hyperparameters
    mlp_epochs: int = 120
    mlp_batch_size: int = 128
    mlp_learning_rate: float = 1e-3
    mlp_patience: int = 15
    mlp_hidden_dim: int = 256
    mlp_latent_dim: int = 64
    mlp_dropout: float = 0.25

    def __post_init__(self) -> None:
        if self.cancer_types is not None:
            self.cancer_types = resolve_cancer_types(self.cancer_types)
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir)
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.outputs_dir = Path(self.outputs_dir)
        self.figure_dir = Path(self.figure_dir)
        self.table_dir = Path(self.table_dir)

    @property
    def rna_path(self) -> Path:
        return self.raw_dir / RNA_FILENAME

    @property
    def cdr_path(self) -> Path:
        return self.raw_dir / CDR_FILENAME

    # Backward-compatible alias used in older notebook cells.
    @property
    def embedding_max_samples(self) -> int | None:
        return self.max_samples_for_embedding

    def ensure_dirs(self) -> None:
        for path in [self.data_dir, self.raw_dir, self.processed_dir, self.outputs_dir, self.figure_dir, self.table_dir]:
            path.mkdir(parents=True, exist_ok=True)
