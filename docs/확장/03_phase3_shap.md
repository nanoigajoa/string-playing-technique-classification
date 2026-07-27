# Phase 3 — SHAP 기반 해석가능성

## 실제 결과 (2026-06-30 완료)

> 설계 문서는 아래. 실행 결과 요약은 여기.
> 계획 대비 변경: 케이스 스터디 4개 → 2개(ordinario, tremolo). Phase 1/2 교집합 없어 Case 1 원안 불가 → ALM 최악 클래스로 대체.

**Feature group 중요도 (차원 수 정규화):**

| 순위 | 그룹 | per-feature mean |SHAP| |
|---|---|---|
| 1 | **Onset-strength envelope** | **0.481** |
| 2 | RMS | 0.046 |
| 3 | Spectral (sc) | 0.029 |
| 4 | MFCC+Δ+ΔΔ | 0.023 |
| 5 | ZCR | 0.011 |
| 6 | Chroma | 0.004 |
| 7 | Log first-onset time | 0.002 |

- `flux_std_0` 방향(실측 median): 낮음 0.18 → col_legno_battuto, 중간 0.51 → ordinario, 높음 0.93 → tremolo
- Surrogate fidelity (XGBoost ↔ MERT, **out-of-sample GroupKFold**): 일치율 **0.914**, Cohen's κ **0.882** — 이전 in-sample 0.998은 암기 착시. 소수 클래스(sul_pont 0.74, col_legno 0.72)에서 대리 신뢰도 낮음
- Phase 2/3 연결은 상관관계 수준 (인과 확인 위한 개입 실험 미수행)
- 결과 파일: `results/phase3/` (report.md, shap_group_bar.png, shap_flux_direction.png, shap_surrogate_fidelity.png, shap_per_class_bar.png, shap_case_*.png)
- 스크립트: `src/phase3_shap.py`

---

## 목적

Phase 1 분류기가 왜 맞히고 왜 틀리는지를 음향학적으로 설명한다.
단독 실험이 아니라 Phase 1/2의 결과를 음향학적으로 해석하는 마지막 층이다.

최종 서사:
**"라벨 부족이 병목이고(Phase 1), 그 병목은 음향학적으로 설명 가능하며(Phase 3),
대형 ALM도 같은 지점에서 같은 이유로 실패한다(Phase 2)."**

---

## 왜 B안인가 (XGBoost SHAP + librosa 직접 연결)

A안(MERT head SHAP + acoustic feature 상관분석)을 선택하지 않은 이유:

MERT 임베딩의 각 차원은 사전학습 과정에서 형성된 추상적 표현 공간이다.
"임베딩 dim 342의 SHAP value가 높다"는 것이 "spectral flux가 중요하다"로 직결되지 않는다.
이 연결을 만들려면 임베딩 차원과 librosa 특징 간의 별도 상관분석이 필요하고,
그 결과가 약한 상관 수준으로 나오면 설명이 오히려 흐려진다.

XGBoost는 v1에서 이미 학습된 모델이고, 245개 특징 각각이 librosa 추출 값이다.
SHAP TreeExplainer가 "이 pizzicato 예측에서 spectral flux mean이 +0.8 기여했다"고 직접 말해준다.
음악 전공자 입장에서 음향학적으로 즉시 해석 가능한 설명이 나온다.

Phase 3의 목표는 "SHAP을 새로 구현하는 것"이 아니라
"Phase 1/2 오류 패턴을 음향학적으로 설명하는 것"이므로 B안이 맞다.

---

## Phase 1/2 교집합 → 케이스 스터디 대상 선정 (단계 3-1)

Phase 1과 Phase 2가 완료되어야 이 단계가 가능하다.

### 공통 실패 클래스 식별

Phase 1에서 "seed 비율 전반에 걸쳐 consistently macro F1이 낮은 클래스":
- 모든 seed 비율(5%~50%)과 τ 조합에서 per-class F1이 평균 이하인 클래스
- `col_legno_battuto`와 `sul_ponticello`가 후보로 예상됨 (v1 결과에서도 어려운 클래스)

Phase 2에서 "오인식률 상위 클래스":
- yes/no와 open-ended 양쪽에서 오인식률이 높은 클래스

교집합을 케이스 스터디 1번 대상으로 우선 선정.

```
예상 교집합: col_legno_battuto (소수 샘플 + 타악적 특성으로 두 모델 모두 어려움)
```

교집합이 비어 있어도 결과다 — "두 모델이 서로 다른 클래스에서 실패한다"는 것도 의미 있는 발견.

---

## XGBoost SHAP 심화 분석 (단계 3-2)

### 전제 조건

v1 노트북(`notebooks/02_modeling_v1.ipynb`)에서 XGBoost 학습이 완료되고
`models/xgb_v1.joblib` 또는 동등한 파일이 저장되어 있어야 한다.
v1 노트북에서 SHAP을 이미 일부 실행했다면 그 결과를 확장한다.

### Explainer 설정

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
# shap_values shape: (n_samples, n_features, n_classes)
# X_test: (n_samples, 245)
# FEATURE_NAMES: src/features.py의 make_feature_names() 결과
```

### 분석 단위: 특징 그룹별 기여도

245차원 개별 특징보다 그룹별로 묶어서 기여도를 보는 것이 해석에 유용하다.

| 그룹 | 차원 수 | 음향학적 의미 |
|---|---|---|
| MFCC (mean+std+q25+q75) | 52 | 음색의 스펙트럼 형태 |
| MFCC Δ (delta) | 52 | 음색의 시간적 변화율 |
| MFCC ΔΔ (delta2) | 52 | 음색 변화율의 가속도 |
| Spectral Contrast | 28 | 배음 vs 잡음 에너지 비율 |
| Onset-strength envelope (`flux_*`) | 4 | mel 기반 half-wave-rectified spectral flux; 프레임간 에너지 변화 |
| ZCR | 4 | 영점 교차율 (노이즈 vs 순음) |
| RMS Energy | 4 | 진폭 크기와 변동 |
| Chroma | 48 | 피치 클래스 분포 (기법 분류에 무효 — 대조군) |
| Log first-onset time | 1 | 첫 onset의 시각 log1p (attack time 아님) |

그룹별 |SHAP value| 합산으로 "그룹 중요도" 계산.

### 시각화 유형

| 시각화 | 목적 | 파일명 |
|---|---|---|
| Beeswarm (전체 클래스) | feature 전반의 영향 방향과 크기 | `shap_beeswarm_overall.png` |
| Beeswarm (클래스별) | 각 기법 예측에서 어떤 특징이 결정적인가 | `shap_beeswarm_{class}.png` |
| Waterfall (케이스별) | 특정 샘플 하나의 예측 과정 | `shap_waterfall_case{n}.png` |
| Bar (그룹별 중요도) | 어떤 특징 그룹이 전반적으로 중요한가 | `shap_group_importance.png` |

---

## 케이스 스터디 4개 (단계 3-3)

각 케이스는 샘플 하나를 중심으로 SHAP waterfall과 librosa 시각화를 결합한다.
케이스 선정은 Phase 1/2 결과가 나온 후 실제 데이터 기반으로 확정.

### Case 1: 두 모델이 동시에 실패하는 샘플

- 조건: Phase 1 XGBoost가 오분류 + Phase 2 ALM이 오인식한 동일 기법 샘플
- SHAP으로 "왜 XGBoost가 틀렸는가" → 해당 특징의 librosa 값을 확인
- 예상 분석: "col_legno_battuto를 ordinario로 오분류한 샘플 — log-attack-time이 ordinario 범주와 겹침"
- 연결: "ALM도 동일 오디오에서 틀렸다면, 이 음향적 모호성은 모델 종류와 무관하게 근본적인 어려움"

### Case 2: Phase 1(XGBoost)은 맞혔지만 Phase 2 ALM이 틀린 샘플

- 조건: XGBoost 정분류 + ALM 오인식
- SHAP으로 "XGBoost가 어떤 특징에 근거해 맞혔는가"
- 이 특징이 ALM의 학습 데이터에 반영되기 어려운 미세한 acoustic 단서인지 확인
- 예상 분석: "spectral flux의 std(표준편차) — 음절 간 에너지 불규칙성 — 이 값이 XGBoost는 포착했지만 ALM은 텍스트 서술로 표현하기 어려운 단서"

### Case 3: Chroma가 오히려 방해한 사례

- 조건: Chroma SHAP value가 양수(기여)인데 예측이 틀린 샘플
- v1 ablation에서 Chroma 제거 시 성능 변화 <1%였지만,
  개별 샘플 수준에서 Chroma가 잘못된 방향으로 기여하는 케이스가 있을 수 있다
- "피치 클래스 정보가 기법 분류를 방해한 구체적 사례"로 framing

### Case 4: Self-training pseudo-label 오류 샘플의 음향적 특성

- 조건: Phase 1에서 높은 confidence(τ > 0.9)로 pseudo-label이 부여됐지만 실제 oracle label과 다른 샘플
- 이 샘플의 SHAP 분석: "모델이 왜 잘못 확신했는가" — 어떤 특징이 잘못된 방향으로 강하게 기여했는가
- "Self-training의 오염이 어떤 음향 조건에서 발생하는가"에 대한 음향학적 설명

---

## 삼각 연결 최종 리포트 (단계 3-4)

`results/phase3/report.md`는 Phase 1/2/3를 하나의 주장으로 연결하는 문서다.

### 구성

**1. 공통 실패 지점 확인**
- Phase 1에서 어려운 클래스
- Phase 2에서 ALM이 틀리는 클래스
- 교집합 시각화

**2. 음향학적 원인 설명 (SHAP 결과 기반)**
- 해당 기법의 결정적 음향 특징이 무엇인가
- 왜 그 특징이 모델에게 어렵게 작용하는가

예상 설명 구조 (col_legno_battuto 가정):
```
col_legno_battuto는 활대 나무 부분으로 현을 타격하는 기법이다.
SHAP 분석에 따르면 spectral flux std와 ZCR mean이 이 기법의 결정적 특징이나,
col_legno_battuto 중 일부 샘플은 이 값이 ordinario 범주와 겹친다.
이는 활대 타격 강도와 음정에 따라 음향 특성이 크게 달라지기 때문이다.
XGBoost는 이 모호한 샘플을 오분류했고, Qwen2-Audio도 동일 샘플에서 틀렸다.
두 모델이 서로 다른 방법론(수동 특징+트리 앙상블 vs 음향 이해 대형 모델)임에도
동일한 지점에서 실패한다는 것은, 이 모호성이 라벨링된 학습 데이터의 부족보다는
기법 자체의 음향학적 이분성(bimodality)에서 기인함을 시사한다.
```

**3. "더 많은 라벨이 도움이 될 클래스 vs 근본적으로 어려운 클래스" 구분**

Phase 1 학습 곡선에서:
- 라벨이 늘수록 꾸준히 F1이 오르는 클래스 → 라벨 부족이 실제 원인
- 50% 라벨에서도 F1이 낮은 클래스 → 음향학적 본질적 어려움

이 구분이 "어떤 기법 분류 태스크에 추가 라벨링 투자가 가치 있는가"라는 실용적 질문에 대한 답이다.

---

## v1 SHAP 결과와의 연속성

v1에서 이미 SHAP beeswarm과 feature importance가 일부 만들어졌다.
Phase 3는 이것의 완전한 재실행이 아니라 **목적 지향적 확장**이다:

| v1에서 이미 한 것 | Phase 3에서 추가하는 것 |
|---|---|
| 전체 특징 중요도 순위 | Phase 1/2 실패 클래스에 집중한 클래스별 SHAP |
| 전반적인 beeswarm | 특정 오분류 샘플의 waterfall (케이스 스터디) |
| Chroma ablation | Chroma가 개별 샘플 수준에서 방해한 케이스 |
| "어떤 특징이 중요한가" | "왜 이 샘플에서 틀렸는가" |

---

## 주요 위험 요소

| 위험 | 대응 |
|---|---|
| Phase 1/2 공통 실패 클래스 교집합이 없는 경우 | "두 모델이 다른 클래스에서 실패한다"는 비교 분석으로 전환 — 보완적 관계도 의미 있는 발견 |
| v1 XGBoost 모델 파일 없는 경우 | 02_modeling_v1.ipynb의 마지막 셀 실행해 `models/xgb_v1.joblib` 재저장 |
| Case 4 (pseudo-label 오류)에 해당하는 샘플 수 부족 | τ=0.7처럼 낮은 threshold에서 오염 샘플을 의도적으로 더 수집해 분석 |
| SHAP 계산 시간 과다 (3,402 샘플) | test set 샘플만 사용 (GroupKFold fold 1 기준), 케이스 스터디는 소수 샘플로 충분 |
