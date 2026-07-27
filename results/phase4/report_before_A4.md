# Phase 4 — Pool B Zero-Shot 평가 결과

## 모델: MERT head (OrchideaSOL 전체 라벨 학습 후 Philharmonia Pool B 적용)

| 지표 | 값 |
|---|---|
| OrchideaSOL 내부 Macro F1 (참고) | 0.9951 |
| **Pool B Cross-Domain Macro F1** | **0.4226** |
| 도메인 이동 손실 | +0.5725 |

## Per-class F1 (Pool B)

```
                   precision    recall  f1-score   support

        ordinario      0.885     0.767     0.821        90
  pizzicato_secco      0.469     0.938     0.625        32
          tremolo      0.516     0.941     0.667        17
   sul_ponticello      0.000     0.000     0.000        13
col_legno_battuto      0.000     0.000     0.000        22

         accuracy                          0.661       174
        macro avg      0.374     0.529     0.423       174
     weighted avg      0.594     0.661     0.605       174

```

## Pool B 구성

| 클래스 | 샘플 수 |
|---|---|
| ordinario | 90 |
| pizzicato_secco | 32 |
| tremolo | 17 |
| sul_ponticello | 13 |
| col_legno_battuto | 22 |

## 시각화

- `poolB_zeroshot_confusion.png`

---
*생성일: Phase 4 단계 4-3*


---

# Phase 4 — Self-Training 결과

## τ × iteration 요약

| τ | Iter | Pool B F1 | 채택률 | Pseudo-label 정확도 |
|---|---|---|---|---|
| 0.7 | 1.0 | 0.4226 | 77.4% | 0.6859 |
| 0.7 | 2.0 | 0.4490 | 87.1% | 0.6752 |
| 0.8 | 1.0 | 0.4226 | 69.2% | 0.7061 |
| 0.8 | 2.0 | 0.4538 | 77.7% | 0.6965 |
| 0.9 | 1.0 | 0.4226 | 53.3% | 0.7116 |
| 0.9 | 2.0 | 0.4286 | 61.3% | 0.6883 |

## 비교

| 조건 | Pool B Macro F1 |
|---|---|
| Zero-shot 기준선 (단계 4-3) | 0.4226 |
| **Self-training 최고 (τ=0.8)** | **0.4610** |
| 향상폭 | +0.0384 |

## Per-class (Self-training τ=0.8)

```
                   precision    recall  f1-score   support

        ordinario      0.890     0.811     0.849        90
  pizzicato_secco      0.478     1.000     0.646        32
          tremolo      0.680     1.000     0.810        17
   sul_ponticello      0.000     0.000     0.000        13
col_legno_battuto      0.000     0.000     0.000        22

         accuracy                          0.701       174
        macro avg      0.410     0.562     0.461       174
     weighted avg      0.615     0.701     0.637       174

```

## 시각화

- `poolB_selftraining_confusion.png`
- `pseudolabel_accuracy_by_class.png`


---

# Phase 4 — Self-Training 결과 (누수 수정본)

## τ 선택 — OrchideaSOL 내부 val (타깃 도메인 라벨 미사용)

> 결함 B 수정: Pool B(holdout)로 τ를 고르던 누수를 제거.
> τ는 소스 도메인(OrchideaSOL) val에서만 결정하고 Philharmonia에 그대로 적용.
> "타깃 도메인 정보 없이 정한 τ가 새 도메인에서 작동하는가"를 측정한다.

| τ | OrchideaSOL val Macro F1 | |
|---|---|---|
| 0.7 | 0.9634 | ← 선택 |
| 0.8 | 0.9634 |  |
| 0.9 | 0.9634 |  |

선택된 τ = **0.7**

## 선택된 τ에서 pseudo-label 채택 (Pool A)

| Iter | 채택 수 | 채택률 | Pseudo-label 정확도 |
|---|---|---|---|
| 1.0 | 312.0 | 77.4% | 0.6859 |
| 2.0 | 351.0 | 87.1% | 0.6752 |

## 비교 (Pool B — 최종 1회 평가)

| 조건 | Pool B Macro F1 |
|---|---|
| Zero-shot 기준선 (단계 4-3) | 0.4226 |
| **Self-training (τ=0.7, 소스 val에서 선택)** | **0.4492** |
| 향상폭 | +0.0266 |

## Per-class (Self-training τ=0.7)

```
                   precision    recall  f1-score   support

        ordinario      0.900     0.800     0.847        90
  pizzicato_secco      0.463     0.969     0.626        32
          tremolo      0.630     1.000     0.773        17
   sul_ponticello      0.000     0.000     0.000        13
col_legno_battuto      0.000     0.000     0.000        22

         accuracy                          0.690       174
        macro avg      0.398     0.554     0.449       174
     weighted avg      0.612     0.690     0.629       174

```

## 시각화

- `poolB_selftraining_confusion.png`
- `pseudolabel_accuracy_by_class.png`
