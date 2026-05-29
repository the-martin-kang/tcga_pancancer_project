# TCGA Pan-Cancer Project v4.3 Patch

## 수정 내용

1. `plot_embedding()` legend marker 누락 문제를 확실히 수정했습니다.
   - PRAD, SARC, STAD, THCA, UCEC처럼 20-class plot의 뒤쪽 label marker가 legend에서 비어 보이던 문제를 해결했습니다.
   - line-only marker를 제거하고 채워진 marker만 사용합니다.
   - Matplotlib 자동 scatter legend 대신 `Line2D` handle로 legend를 직접 생성합니다.

2. 노트북 import cell에 `importlib.reload()`를 추가했습니다.
   - 패치 파일을 덮어쓴 뒤 kernel을 완전히 재시작하지 않은 경우에도 구버전 함수가 남을 가능성을 줄였습니다.
   - 그래도 가장 안전한 방법은 파일 교체 후 `Kernel > Restart`입니다.

3. v4.2의 feature-set comparison tqdm progress 기능은 유지했습니다.

## 적용 방법

기존 프로젝트 폴더에서 아래 파일만 교체하세요.

```text
tcga_project/visualization.py
tcga_project/experiments.py
TCGA_PanCancer_Project_v4_3.ipynb
```

기존 PNG는 자동으로 수정되지 않습니다. 아래 section을 다시 실행해야 새 legend가 반영됩니다.

```text
6. 모델링 전 차원축소 시각화
14. MLP latent representation 분석
```

## 확인

`v43_legend_marker_test.png`에서 20개 class 모두 legend marker가 표시되는 것을 확인했습니다.
