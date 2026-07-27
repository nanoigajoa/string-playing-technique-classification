# Phase 4 — Pool B Zero-Shot 평가 결과

## 모델: MERT head (OrchideaSOL 전체 라벨 학습 후 Philharmonia Pool B 적용)

| 지표 | 값 |
|---|---|
| OrchideaSOL in-domain **LOSO** Macro F1 (기준) | 0.8925 |
| **Pool B Cross-Domain Macro F1** | **0.4226** |
| **도메인 이동 손실 (LOSO 대비)** | **+0.4698** |
| — in-sample 재예측 (낙관 착시, 기준 아님) | 0.9951 |

> **A4 수정**: 도메인 손실은 in-domain **LOSO**(보지 못한 악기로의 일반화)를 기준으로 잰다.
> 이전 판은 in-sample 재예측 0.9951를 기준으로 써 손실을 +0.5725로
> 부풀렸다. 정직한 손실은 **+0.4698**이다.

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

## ⚠️ 철회된 결과 — Self-Training τ=0.8 → Pool B 0.4610

> **이 수치는 철회되었다. 인용 금지.** 초기 self-training 실험은 τ∈{0.7,0.8,0.9}를
> **Pool B(타깃 holdout) F1로 골라** τ=0.8에서 Pool B 0.4610(+0.0384)을 보고했다.
> 이는 함정①(선택-시점 누수) — 평가셋으로 하이퍼파라미터를 고른 뒤 같은 셋에서 성능을
> 보고한 것 — 그 자체다. 누수를 제거하고 τ를 **소스 도메인(OrchideaSOL) val**에서만 고르니
> τ=0.7이 선택되고 Pool B는 **0.4492(+0.0266)**로 내려갔다(아래 누수 수정본).
> 이 철회 사례 자체가 본 논문 함정①의 cross-domain 변형 증거로 사용된다
> (`docs/method_leakage_note.md`).

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
