# TCGA Pan-Cancer Project v4.2 patch

Modified files only.

## Files
- `TCGA_PanCancer_Project_v4_2.ipynb`
- `tcga_project/visualization.py`
- `tcga_project/experiments.py`

## Changes
1. Fixed missing legend/points for PRAD, SARC, STAD, THCA, UCEC in all embedding plots.
   - Cause: v4 used line-only markers (`+`, `x`, `1`, `2`, `3`) with `linewidths=0.0`.
   - Fix: all 20-class markers are now filled markers.

2. Added tqdm-based progress display for feature-set size comparison.
   - Overall tqdm across feature-set specs.
   - Stage tqdm for preprocess / fit / evaluate.
   - Alive tqdm while long scikit-learn `LogisticRegression.fit()` is running.

## Apply
Copy the two Python files into your existing project folder:

```text
tcga_project/visualization.py
tcga_project/experiments.py
```

Then open `TCGA_PanCancer_Project_v4_2.ipynb` and restart the kernel before running.
