# Phase 1 Results Summary

CLAP zero-shot baseline : 0.3630
v1 SVM baseline         : 0.916

> 누수 수정: iteration은 val_select F1로 선택(test 미사용). 전 수치 4-fold 평균±표준편차, 단일 fold 헤드라인 제거.

## Baseline vs Self-Training (mean over folds x trials)

|             |   baseline_f1 |   best_st_f1 |
|:------------|--------------:|-------------:|
| (0.05, 0.7) |        0.7259 |       0.6381 |
| (0.05, 0.8) |        0.7259 |       0.6192 |
| (0.05, 0.9) |        0.7259 |       0.6641 |
| (0.1, 0.7)  |        0.7254 |       0.7392 |
| (0.1, 0.8)  |        0.7254 |       0.7267 |
| (0.1, 0.9)  |        0.7254 |       0.718  |
| (0.2, 0.7)  |        0.7767 |       0.7977 |
| (0.2, 0.8)  |        0.7767 |       0.7784 |
| (0.2, 0.9)  |        0.7767 |       0.7596 |
| (0.5, 0.7)  |        0.821  |       0.8294 |
| (0.5, 0.8)  |        0.821  |       0.8363 |
| (0.5, 0.9)  |        0.821  |       0.8244 |

## Key Findings (fold-level 평균±표준편차, n=4)

- seed 5%: best τ=0.9, self-training F1=0.6641±0.0484 (fold-level, n=4)
- seed 10%: best τ=0.7, self-training F1=0.7392±0.0534 (fold-level, n=4)
- seed 20%: best τ=0.7, self-training F1=0.7977±0.0347 (fold-level, n=4)
- seed 50%: best τ=0.8, self-training F1=0.8363±0.0316 (fold-level, n=4)
- MERT head (seed=50%): F1=0.8210±0.0452 (below v1 SVM 0.916)
- Self-training vs baseline 향상폭의 통계적 유의성은 `phase1_significance.py` 참조 (fold n=4, 검정력 한계 있음)
