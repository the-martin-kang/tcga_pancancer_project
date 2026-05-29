# TCGA Pan-Cancer RNA-seq Cancer Type Classification

> **TCGA Pan-Cancer RNA-seq 기반 암종 분류와 해석 가능한 바이오마커 후보 탐색**  
> Classical Machine Learning과 MLP latent representation을 비교하는 end-to-end 데이터 과학 프로젝트

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-red)
![Dataset](https://img.shields.io/badge/Data-TCGA%20PanCanAtlas-green)

---

## 1. Project Overview

이 프로젝트는 **TCGA Pan-Cancer Atlas RNA-seq gene expression 데이터**를 이용해 여러 암종을 분류하는 데이터 과학 파이프라인입니다.
단순히 높은 정확도의 모델을 만드는 것보다, 수업에서 배운 전체 데이터 분석 절차를 따라 아래 과정을 하나의 노트북과 Python 모듈로 정리하는 것을 목표로 합니다.

```text
문제 정의
→ 데이터 수집
→ 데이터 검사 및 EDA
→ 결측치 처리와 전처리
→ feature selection
→ classical ML / MLP 모델링
→ 평가
→ 해석 가능한 gene-level 분석
→ PCA / UMAP / t-SNE 기반 시각화
```

메인 노트북은 핵심 실행 흐름만 보여주고, 데이터 로딩·전처리·시각화·모델링 함수는 `tcga_project/` 패키지로 분리했습니다.

---

## 2. Research Questions

이 프로젝트에서 다루는 핵심 질문은 다음과 같습니다.

1. **RNA-seq gene expression profile만으로 암종을 분류할 수 있는가?**
2. **Logistic Regression, Random Forest, MLP는 성능과 해석 가능성 측면에서 어떤 차이를 보이는가?**
3. **PCA, UMAP/t-SNE, MLP latent representation은 암종 간 구조를 시각적으로 잘 보여주는가?**
4. **모델이 중요하게 사용한 gene들은 암종별 candidate biomarker로 해석할 수 있는가?**
5. **Top 2,000 variable genes만 사용하는 경우와 전체 usable genes를 사용하는 경우 성능 차이가 있는가?**

---

## 3. Dataset

### Main dataset

- **Dataset**: TCGA Pan-Cancer Atlas / PanCanAtlas
- **Main input**: RNA-seq gene expression matrix
- **File used in code**: `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv`
- **Label source**: TCGA cancer type annotation from TCGA-Clinical Data Resource / sample metadata
- **Task type**: Multi-class cancer type classification

TCGA는 National Cancer Institute와 National Human Genome Research Institute가 수행한 대규모 암 유전체 프로그램이며, 33개 암종에 걸친 20,000개 이상의 primary cancer 및 matched normal sample을 분자적으로 분석한 공개 자원입니다.

### Data access

Raw data는 repository에 포함하지 않습니다. 노트북을 실행하면 공식 GDC PanCanAtlas supplemental data에서 필요한 파일을 `data/raw/`에 다운로드하도록 구성되어 있습니다.

References:

- [The Cancer Genome Atlas Program - NCI](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)
- [TCGA PanCanAtlas Publications - GDC](https://gdc.cancer.gov/about-data/publications/pancanatlas)

---

## 4. Project Structure

```text
.
├── TCGA_PanCancer_Project_v4_3.ipynb      # Main notebook
├── README.md                              # Project overview
├── requirements.txt                       # Python dependencies
├── tcga_project/
│   ├── config.py                          # Project configuration and cancer type metadata
│   ├── data.py                            # Data download, loading, barcode parsing
│   ├── preprocessing.py                   # Missing value handling, feature selection, scaling
│   ├── models.py                          # Classical ML models and evaluation
│   ├── deep_learning.py                   # PyTorch MLP model
│   ├── dimensionality.py                  # PCA, UMAP, t-SNE, clustering utilities
│   ├── interpretation.py                  # Logistic coefficients and feature importance
│   ├── visualization.py                   # Plotting functions
│   ├── experiments.py                     # Feature-set comparison experiment
│   └── utils.py                           # Helper functions
├── data/
│   ├── raw/                               # Downloaded raw TCGA files; not tracked
│   └── processed/                         # Optional intermediate files; not tracked
└── outputs/
    ├── figures/                           # Generated plots
    └── tables/                            # Generated CSV summaries
```

---

## 5. Main Analysis Pipeline

### 5.1 Data Collection

The notebook downloads or loads TCGA PanCanAtlas RNA-seq expression data and metadata.

Main processing steps:

- Load gene expression matrix
- Parse TCGA sample barcodes
- Merge expression samples with cancer type labels
- Select primary tumor samples
- Select top 20 cancer types or user-defined cancer type list

Default project setting:

```python
CANCER_TYPES = None
N_TOP_CANCER_TYPES = 20
MAX_SAMPLES_PER_CLASS = 1000
```

With these settings, the notebook automatically selects the 20 most frequent cancer types after filtering.

---

### 5.2 EDA and Data Inspection

The project checks basic dataset quality before modeling.

Generated summaries include:

- Number of samples
- Number of genes
- Number of cancer classes
- Class imbalance
- Missing value count and missing fraction
- Expression distribution
- PCA / UMAP or t-SNE visualization

Example output files:

```text
outputs/tables/dataset_summary.csv
outputs/tables/missing_value_report.csv
outputs/tables/class_counts.csv
outputs/figures/class_counts.png
outputs/figures/expression_distribution.png
outputs/figures/eda_pca_raw_expression.png
outputs/figures/eda_umap_raw_expression.png
```

---

### 5.3 Preprocessing

Preprocessing is designed to avoid data leakage.

Important rule:

> Imputation, feature selection, and scaling are fitted only on the training set, then applied to the test set.

Main steps:

1. Train/test split with stratification
2. Remove genes with too many missing values
3. Median imputation by gene
4. Low-variance gene removal
5. Top variable gene selection
6. Standard scaling

Default feature setting:

```python
TOP_N_GENES = 2000
MAX_MISSING_FRACTION = 0.20
```

---

### 5.4 Modeling

This project compares simple classical ML models and an MLP model.

| Model | Role |
|---|---|
| Dummy majority classifier | Baseline model |
| Logistic Regression L2 | Stable linear baseline |
| Logistic Regression L1 | Sparse and interpretable gene selection |
| Random Forest | Nonlinear classical ML baseline |
| MLP | Neural network classifier and latent representation extractor |

---

### 5.5 Evaluation

Because cancer type classes are imbalanced, the project does not rely only on accuracy.

Metrics:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- One-vs-rest macro ROC-AUC
- Row-normalized confusion matrix

Raw count confusion matrices can be misleading when class sizes are imbalanced. Therefore, the main confusion matrix figure uses **row normalization**, meaning each true class row sums to 1.

Example output files:

```text
outputs/tables/all_model_metrics.csv
outputs/tables/classical_model_metrics.csv
outputs/tables/mlp_metrics.csv
outputs/tables/confusion_matrix_counts_RandomForest.csv
outputs/tables/confusion_matrix_row_normalized_RandomForest.csv
outputs/figures/confusion_matrix_row_normalized_RandomForest.png
outputs/figures/per_class_recall_RandomForest.png
outputs/figures/top_confusion_pairs_RandomForest.png
```

---

### 5.6 Interpretation

The project extracts candidate marker genes from trained models.

Interpretation methods:

- Class-wise Logistic Regression coefficients
- L1 Logistic Regression nonzero genes
- Random Forest feature importance
- Overlap between important gene lists

These genes should be interpreted as **model-based candidate marker genes**, not experimentally validated biomarkers.

Example output files:

```text
outputs/tables/top_genes_logreg_l2.csv
outputs/tables/nonzero_genes_logreg_l1.csv
outputs/tables/top_genes_random_forest.csv
outputs/tables/overlapping_top_genes.csv
outputs/figures/top_genes_logreg_BLCA.png
```

---

### 5.7 Dimensionality Reduction and Latent Representation

For visual interpretation, the project compares raw expression space and MLP latent space.

Methods:

- PCA
- UMAP, if `umap-learn` is installed
- t-SNE fallback
- K-means clustering on reduced or latent space

The visualization module uses a 20-class categorical palette and custom legend handles so that all 20 cancer types are displayed clearly.

Example output files:

```text
outputs/figures/cancer_type_palette_preview.png
outputs/figures/eda_pca_raw_expression.png
outputs/figures/eda_umap_raw_expression.png
outputs/figures/mlp_training_history.png
```

---

### 5.8 Feature Set Size Comparison

The notebook includes an optional experiment comparing:

1. Top 2,000 variable genes
2. All usable genes after missing-value and variance filtering

This section can take longer than the main model training, so `tqdm` progress bars are used.

```python
RUN_FEATURE_SET_COMPARISON = True
FEATURE_SET_SPECS = [
    {"name": "Top 2,000 variable genes", "top_n_genes": 2000},
    {"name": "All usable genes", "top_n_genes": None},
]
```

Generated files:

```text
outputs/tables/feature_set_size_comparison_logreg.csv
outputs/figures/feature_set_size_comparison_logreg.png
```

If this section is too slow, set:

```python
RUN_FEATURE_SET_COMPARISON = False
```

---

## 6. Example Results

The following example values are from one local v4 run using 20 cancer types and up to 1,000 samples per class.

### Dataset summary

| Item | Value |
|---|---:|
| Samples | 8,459 |
| Genes | 20,501 |
| Classes | 20 |
| Missing fraction | 1.73% |

### Model performance

| Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC OvR Macro |
|---|---:|---:|---:|---:|
| Random Forest | 0.9645 | 0.9551 | 0.9577 | 0.9989 |
| Logistic Regression L1 | 0.9604 | 0.9584 | 0.9546 | 0.9975 |
| MLP | 0.9592 | 0.9566 | 0.9536 | 0.9984 |
| Logistic Regression L2 | 0.9598 | 0.9530 | 0.9526 | 0.9980 |
| Dummy majority | 0.1182 | 0.0500 | 0.0106 | 0.5000 |

These numbers may change depending on random seed, selected cancer types, maximum samples per class, and preprocessing settings.

---

## 7. Installation

### Option 1: venv

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda create -n tcga-pancancer python=3.10 -y
conda activate tcga-pancancer
pip install -r requirements.txt
```

Recommended packages:

```text
numpy
pandas
scikit-learn
matplotlib
openpyxl
requests
tqdm
torch
umap-learn
```

---

## 8. How to Run

1. Clone this repository.
2. Install dependencies.
3. Open the notebook.
4. Run cells from top to bottom.

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
jupyter lab TCGA_PanCancer_Project_v4_3.ipynb
```

The first run may take time because the TCGA data file is downloaded and parsed.

If the raw data has already been downloaded in another project folder, copy it into:

```text
data/raw/
```

---

## 9. Important Configuration Options

Most user-adjustable settings are located near the top of the notebook.

```python
CANCER_TYPES = None
N_TOP_CANCER_TYPES = 20
MAX_SAMPLES_PER_CLASS = 1000
TOP_N_GENES = 2000
MAX_MISSING_FRACTION = 0.20
EMBEDDING_METHOD = "umap"
MAX_SAMPLES_FOR_EMBEDDING = 3000
RUN_MLP = True
RUN_FEATURE_SET_COMPARISON = True
```

Suggested quick-run settings:

```python
MAX_SAMPLES_PER_CLASS = 200
TOP_N_GENES = 1000
RUN_MLP = False
RUN_FEATURE_SET_COMPARISON = False
```

Suggested full-run settings:

```python
MAX_SAMPLES_PER_CLASS = 1000
TOP_N_GENES = 2000
RUN_MLP = True
RUN_FEATURE_SET_COMPARISON = True
```

---

## 10. Notes and Limitations

- This project is an educational machine learning analysis, not a clinical diagnostic tool.
- Model-based important genes are candidate markers and require biological and experimental validation.
- Batch effects and tissue-of-origin effects can strongly influence pan-cancer expression patterns.
- External validation on independent datasets such as GEO is not included in the main pipeline.
- The optional all-gene comparison can be computationally expensive.

---

## 11. Version Notes

### v4.3

- Fixed 20-class dimensionality-reduction legend rendering.
- Replaced problematic line-only markers with stable filled markers.
- Used custom legend handles to make all cancer types visible.
- Added progress display for feature-set comparison using `tqdm`.

### v4.2

- Added feature-set comparison progress indicators.
- Improved plotting functions for 20 cancer classes.

### v4

- Added 20-class cancer classification mode.
- Added row-normalized confusion matrix.
- Added top 2,000 genes vs all usable genes comparison.
- Added UMAP/t-SNE dimensionality-reduction visualizations.

---

## 12. License and Data Usage

This repository is intended for coursework and educational use.
TCGA/GDC data should be used according to the policies and terms of the Genomic Data Commons and TCGA data access guidelines.

---

## 13. Acknowledgements

- The Cancer Genome Atlas Program, National Cancer Institute
- NCI Genomic Data Commons
- TCGA Pan-Cancer Atlas Research Network
- scikit-learn, PyTorch, pandas, NumPy, matplotlib, UMAP
