# 현악기 연주기법 분류 — 프로젝트 요약

> 최종 갱신: 2026-07-21 (B6 클러스터 부트스트랩 — 함정 3 신설, "동급" 결론 유지)

OrchideaSOL 데이터셋으로 현악기(Violin / Viola / Violoncello / Contrabass)의 5가지 연주기법을
**v1: 특징 공학 + 전통 ML + SHAP**, **v2: Mel-spectrogram + CNN**, 그리고
**Phase 1~4: foundation model + ALM 환각 평가 + SHAP 심화 해석 + 교차 도메인 일반화 검증**으로 확장한 MIR 포트폴리오 프로젝트.

---

## 핵심 내러티브 (2026-07 최종)

> "저차원 강판별 신호(onset-strength envelope)가 존재하는 소규모 세밀분류 태스크에서는,
> **공정하게 평가하면 값싼 수동특징 SVM과 대형 SSL(MERT)이 오차범위 내 동급**이다.
> 이 결론에 이르기까지 self-training 평가의 **두 함정(test-set 누수 + 백본 언더튜닝)**을
> 발견·정량화했다 — 각 함정이 결론을 한 번씩 뒤집었다.
> 이후 **세 번째 함정(불확실성의 재표집 단위)**을 발동 직전에 포착했다."

**결론이 두 번 뒤집힌 궤적** (이 프로젝트의 뼈대):

| 국면 | 관측 | 원인 | 진위 |
|---|---|---|---|
| 착각 ① | "저라벨에서 MERT 유리" | self-training test 누수 | 거짓 |
| 착각 ② | "SVM 전 구간 완승" | MERT 백본 언더튜닝(단일 linear head) | 거짓 |
| **진실** | **공정 튜닝 시 SVM ≈ MERT** | 두 함정 제거 후 | 확정 |
| 착각 ③ *(미수)* | "클러스터 부트스트랩 CI로 5%·20%에서 SVM 유의 우세" | 재표집 단위 오설정(fold 변동 누락) | 거짓 — 포착 |

**함정 3 — 재표집 단위 (B6, 2026-07-21)**: 주 비교 MERT−SVM에 B=2000 부트스트랩 CI.

| seed | 차이 | 1수준(클러스터만, BCa) | 2수준(fold 포함) |
|---|---|---|---|
| 5% | −0.027 | [−0.044, −0.012] **0 제외** | [−0.107, +0.050] 0 포함 |
| 10% | −0.000 | [−0.016, +0.015] | [−0.071, +0.069] |
| 20% | −0.026 | [−0.043, −0.011] **0 제외** | [−0.074, +0.035] 0 포함 |
| 50% | −0.007 | [−0.021, +0.006] | [−0.046, +0.025] |

> recording 클러스터로만 재표집하면 test 클립 변동만 담겨 CI가 2.6~5.0배 좁아지고, 5%·20%에서
> "SVM 유의 우세"라는 §진실과 충돌하는 결론이 나온다. 일반화 주장의 단위가 **악기**이므로
> fold까지 재표집해야 하며, 그러면 **4개 구간 전부 0을 포함** → **"동급" 결론 유지**.
> fold-std(n=4) 분석과 독립적으로 일치. (`results/phase1/cluster_bootstrap.md`)

> 📌 방법론 상세: [method_leakage_note.md](method_leakage_note.md) · 궤적 전체: [../연구과정/00_흐름정리.md](../연구과정/00_흐름정리.md)

보조 발견: 대형 ALM은 미세 주법에서 한계를 보이고(Phase 2), 그 음향학적 근거는 onset-strength envelope로
설명되며(Phase 3), 교차 도메인에서 일부 클래스는 도메인 이동에 취약하다(Phase 4).

---

## 전체 모델 성능 비교

| 모델 | Macro F1 | 평가 방식 | 비고 |
|---|---|---|---|
| **SVM (RBF, v1)** | **0.916** ±0.011 | LOSO 전체라벨 | v1 대표 성능 |
| CNN (SmallCNN, v2) | 0.905 ±0.030 | LOSO 전체라벨 | |
| XGBoost (v1) | 0.893 ±0.032 | LOSO 전체라벨 | SHAP 해석 주 대상 |
| Random Forest (v1) | 0.886 ±0.060 | LOSO 전체라벨 | |
| CLAP zero-shot | 0.363 | 전체 데이터 | 학습 없음, 하한 기준선 |
| Qwen2-Audio Y/N | 0.504 | 125 샘플 | 랜덤 수준 |
| Qwen2-Audio open-ended | 0.200 | 125 샘플 | 5-class 랜덤 수준 |
| MERT cross-domain zero-shot | 0.423 | Philharmonia Pool B | OrchideaSOL → Philharmonia |
| MERT cross-domain + self-training | 0.449 | Philharmonia Pool B | τ=0.7(소스 val 선택), +2.7%p |

평가: **Leave-One-Instrument-Out** (GroupKFold n=4, 악기 단위 완전 분리)

**공정 비교 (seed별, A1 재실행 — 양쪽 val 분리 + 동일 학습데이터 tr-only)** — SVM vs MERT는 전 구간 오차범위 내 동급:

| seed | SVM (튜닝) | MERT (튜닝) | 판정 |
|---|---|---|---|
| 5% | 0.661 | 0.671 | 겹침 (MERT +0.010) |
| 10% | 0.754 | 0.748 | 겹침 (SVM +0.006) |
| 20% | 0.832 | 0.807 | SVM 소폭 (~0.7σ) |
| 50% | 0.868 | 0.859 | 겹침 (SVM +0.009) |

> A1(감사 코드 val 이중사용 제거) 후 값. SVM·MERT를 **동일한 tr(labeled의 64%)**로 학습한
> 주 수치다. SVM이 학습량 이점(seed_core 80%)을 살리면 전 구간 소폭 앞서지만 여전히 ~1σ 이내
> (`results/phase1/mert_tuning_audit.md`). 이전 판 pre-A1 수치(SVM 0.709 / MERT 0.697 등)는 철회.

**관행 probing까지 적용 후 (B2, 레이어 가중)** — MERT에 관행(전 층 학습형 가중)을 적용한 재검증:

| seed | MERT(layer-weighted) | SVM(동일 학습데이터) | 차이 | SVM(seed_core 80%) | 차이 |
|---|---|---|---|---|---|
| 5% | 0.737 | 0.661 | +0.076 (1.25σ) | 0.702 | +0.035 |
| 10% | 0.797 | 0.754 | +0.043 (0.63σ) | 0.799 | −0.002 |
| 20% | 0.855 | 0.832 | +0.024 (0.68σ) | 0.835 | +0.020 |
| 50% | 0.900 | 0.868 | +0.032 (1.01σ) | 0.880 | +0.020 |

> 마지막-층-only는 MERT를 전 구간 ~0.04 언더튜닝했다(학습된 가중 최대 L4=0.126, 마지막 층 L12=0.046).
> 관행 적용 후 방향은 4/4 MERT 쪽이나 **어느 구간도 유의하지 않고**, 동등 학습량을 주면 소멸 →
> **"동급" 결론 유지.** 부수 발견: **probe 설정 효과(+0.04, 일관) > 모델 계열 효과(−0.002~+0.035, 부호 불일치)**.
> (`results/phase2/layer_probe.md`)

> ⚠️ 이전 판(누수·언더튜닝)의 "MERT+ST 단일 fold 0.920", "τ=0.8 cross-domain 0.461" 수치는 **철회**됨.
> 상세: [method_leakage_note.md](method_leakage_note.md), `results/누수제거_전후비교.md`, `results/phase1/mert_tuning_audit.md`

---

## 데이터

총 **3,402개** 샘플 — ordinario 1,193 / tremolo 1,041 / pizzicato_secco 639 / sul_ponticello 302 / col_legno_battuto 227

---

## Phase 1 — MERT Self-Training + CLAP 제로샷 기준선

**목적**: 라벨 비율에 따른 성능 변화 실증, foundation model vs 수동 특징 **공정** 비교

**설정**: MERT-v1-95M 임베딩(768d) + head, seed 5%/10%/20%/50%, τ ∈ {0.7, 0.8, 0.9}
누수 수정: iteration은 **검증셋**으로 선택(test 미사용). 아래는 누수 제거본(τ=0.7).

| seed 비율 | CLAP 기준선 | MERT baseline | MERT + self-training (τ=0.7) |
|---|---|---|---|
| 5% | 0.363 | 0.726 | 0.638 |
| 10% | 0.363 | 0.725 | 0.739 |
| 20% | 0.363 | 0.777 | 0.798 |
| 50% | 0.363 | 0.821 | 0.829 |

**핵심 발견 (누수 제거 후)**:
- Self-training 이득은 **미미**(10~50% +0.01~0.02)하며, **5%에서는 오히려 baseline보다 하락** — 라벨이 적으면 반복 회차 선택 자체가 불안정. (누수판의 "전 구간 향상 +7~8%p"는 test 누수 산물이었음)
- 향상폭 유의성: seed 20% +0.021±0.018, Wilcoxon p=0.125 → **fold n=4 한계로 유의성 확정 불가** (`results/phase1/significance.md`)
- **공정 튜닝 후 MERT ≈ SVM**: 마지막-층 단일 linear head가 MERT를 언더튜닝했음이 확인됨(최적 head는 라벨량에 따라 갈림 — 고라벨 mlp256, 저라벨 linear). 상세 [method_leakage_note.md](method_leakage_note.md)
- 결과 파일: `results/phase1/`, `results/누수제거_전후비교.md`

---

## Phase 2 — Qwen2-Audio 환각(Hallucination) 평가

**목적**: 대형 ALM이 주법처럼 전문가 라벨링이 필요한 미세 영역에서 보이는 한계 실증

**설정**: Qwen2-Audio-7B-Instruct, 125 샘플(25/class), Y/N + open-ended = 250 쿼리

| 질문 유형 | 정확도 | 비고 |
|---|---|---|
| Yes/No | 50.4% | 동전 던지기 수준 |
| Open-ended | 20.0% | 5-class 랜덤 수준 |

**클래스별 오인식률**:

| 클래스 | Y/N miss | Open miss |
|---|---|---|
| ordinario | 84% | 100% |
| pizzicato_secco | 48% | **0%** — 유일하게 open 정확 |
| tremolo | 36% | 100% |
| sul_ponticello | 36% | 100% |
| col_legno_battuto | 44% | 100% |

**Phase 1/2 교집합**: 없음 — MERT 최악(sul_p/col_l) vs ALM 최악(ordinario/tremolo) — 두 모델이 서로 다른 클래스에서 실패

- 결과 파일: `results/phase2/`

---

## Phase 3 — SHAP 해석가능성

**목적**: XGBoost v1 SHAP으로 Phase 2 ALM 실패의 음향학적 원인 분석

**Feature group 중요도** (차원 수 정규화 후):

| 순위 | 그룹 | n | per-feature mean |SHAP| |
|---|---|---|---|
| 1 | **Onset-strength envelope** | 4 | **0.481** |
| 2 | RMS | 4 | 0.046 |
| 3 | Spectral (sc) | 28 | 0.029 |
| 4 | MFCC+Δ+ΔΔ | 156 | 0.023 |
| 5 | ZCR | 4 | 0.011 |
| 6 | Chroma | 48 | 0.004 |
| 7 | Log first-onset time | 1 | 0.002 |

**방향성 발견** (`flux_std_0`): 낮은 값(median 0.18) → col_legno_battuto, 중간 값(0.51) → ordinario, 높은 값(0.93) → tremolo

**Surrogate fidelity** (XGBoost ↔ MERT 예측 일치, **out-of-sample GroupKFold**): 일치율 **0.914**, Cohen's κ **0.882**. 이전에 보고한 in-sample 0.998은 xgb_v1이 학습셋을 암기(정확도 1.0)해 생긴 착시였다. 소수 클래스(sul_ponticello κ 0.74, col_legno κ 0.72)에서 대리 신뢰도가 낮아 그 두 클래스의 SHAP 해석은 MERT로 조심해서만 전이된다.

**Phase 2/3 연결** (상관관계, 인과관계 미확인):
- pizzicato: MFCC delta 높음 + RMS 낮음 → 음향적으로 뚜렷 → ALM도 100% 정확
- ordinario: flux_std 중간값(0.51) → col_legno(0.18) / sul_ponticello(0.48)와 범위 겹침 → ALM 0% (가설)
- tremolo: flux_q75 높음 → 시간 변화 패턴 중요 → ALM이 temporal 패턴 미포착 (가설, 개입 실험 미수행)

- 결과 파일: `results/phase3/`

---

## Phase 4 — 교차 도메인 일반화 검증 (Philharmonia)

**목적**: Phase 1 self-training이 같은 데이터 내 시뮬레이션에 그치지 않음을 독립 도메인으로 검증

**설정**: Philharmonia Orchestra Sound Samples (CC BY-SA 3.0), 577개 (ordinario 300 상한)
→ Pool A 403개(pseudo-label용) / Pool B 174개(holdout), MERT head.
누수 수정: τ는 **OrchideaSOL 소스 val**에서 선택(타깃 정보 미사용) → τ=0.7. Pool B는 최종 1회만 평가.

| 조건 | Pool B Macro F1 |
|---|---|
| Zero-shot (OrchideaSOL → Philharmonia) | 0.4226 |
| **Self-training (τ=0.7, 소스 val 선택)** | **0.4492** (+2.7%p) |

**핵심 발견**:
- 도메인 이동 손실: in-domain **LOSO 0.8925** → cross-domain zero-shot 0.4226 = **−0.47**. (기준을 in-sample 재예측 0.9951로 잡으면 손실이 −0.57로 부풀려짐 — LOSO가 정직한 기준)
- 절대 성능은 낮음(0.45) — cross-domain은 여전히 어렵다. (누수판 "τ=0.8 0.461, +3.8%p"는 holdout 튜닝 산물이라 철회)
- col_legno_battuto, sul_ponticello: self-training 후에도 F1≈0.000 — **도메인 이동에 구조적 취약** (이 발견은 유지)
- col_legno pseudo-label 채택 0% — 모델이 한 번도 확신 있게 예측하지 못함
- **Phase 3 연결**: onset-strength envelope 경계가 이 두 클래스에서 도메인 간 유지되지 않음을 시사
- 결과 파일: `results/phase4/`

---

## 선행 연구와의 차이 (Kruger & Jacobs, 2020, JNMR)

- 다른 데이터셋(OrchideaSOL) + 더 엄격한 LOSO 검증 (논문은 random 70/30 split)
- SVM / XGBoost / RF에 **CNN + MERT self-training + ALM 평가** 추가
- SHAP 그룹 중요도(차원 정규화) + 방향성 분석 + surrogate fidelity 검증

---

## v1/v2 핵심 발견

- LOSO 환경에서 **SVM이 CNN보다 정확(0.916 vs 0.905)하고 안정적(±0.011 vs ±0.030)**
- **Onset-strength envelope**가 단일 feature 기준 압도적 1위 (per-feature SHAP)
- **Chroma** ablation 변화 <1%, 정규화 후 순위 6/7 — 기법 분류에 무효 확인
- **col_legno_battuto**가 전통 ML에서 가장 어려운 클래스 (소수 샘플 + 타악적 어택)

---

저장소: github.com/nanoigajoa/string-playing-technique-classification
