# TCGA Pan-Cancer RNA-seq Classification Project v4

이 프로젝트는 TCGA Pan-Cancer Atlas RNA-seq 데이터를 이용해 암종 분류, 모델 평가, candidate marker gene 해석, MLP latent representation 분석을 수행하는 수업용 end-to-end 데이터 과학 파이프라인입니다.

## v4 변경사항

- 20개 암종 plot에서 matplotlib 기본 10색 cycle이 반복되는 문제를 수정했습니다.
  - `tab20` 기반 categorical palette와 marker cycle을 사용합니다.
  - `outputs/figures/cancer_type_palette_preview.png`에서 색상 매핑을 확인할 수 있습니다.
- Feature selection 규모 비교 section을 추가했습니다.
  - `Top 2,000 variable genes` vs `All usable genes`
  - 동일한 Logistic Regression 기준으로 accuracy, balanced accuracy, macro-F1, runtime을 비교합니다.
- Confusion matrix는 row-normalized plot을 기본 발표용 그림으로 사용합니다.

## 실행

```bash
pip install -r requirements.txt
jupyter notebook TCGA_PanCancer_Project.ipynb
```

처음 실행하면 TCGA PanCanAtlas supplemental data를 `data/raw/`에 다운로드합니다.
