# 연구적 위치 평가 및 발전 방향

> 이 문서는 본 프로젝트를 처음 검토하시는 분(지도교수님 등)을 위해 (1) 저장소 읽는 순서, (2) 논문으로서의 현실성 평가, (3) 아이디어의 출처가 된 선행 연구를 정리한 것입니다.

---

## 1. 저장소 읽는 순서

1. **`README.md`** — 전체 요약, 결과 표, 선행 연구와의 차이점
2. **`docs/00_overview.md` ~ `docs/07_v2_results.md`** — 설계 의도(왜 이렇게 했는가) + 단계별 결과
3. **`notebooks/01_eda.ipynb`, `02_modeling_v1.ipynb`, `03_modeling_v2.ipynb`** — 실행 가능한 재현 코드
4. **본 문서 (`08_research_context.md`)** — 학술적 위치와 참고문헌

`src/`는 노트북에서 사용하는 모듈(특징 추출, 데이터셋, 모델 정의, 추론), `results/`와 `models/`는 각 단계의 산출물입니다.

---

## 2. 논문으로서의 현실성

### 2.1 연구 문제로서의 위치

가장 직접적인 비교 대상은 **Kruger & Jacobs (2020, Journal of New Music Research)** — 현악기 보잉 아티큘레이션을 SVM으로 분류한 연구입니다.

| 항목 | Kruger & Jacobs (2020) | 본 프로젝트 |
|---|---|---|
| 데이터셋 | RWC database | OrchideaSOL (다른 데이터셋) |
| 검증 방법 | random 70/30 split × 20회 | **LOSO GroupKFold** (n=4, 악기 단위 완전 분리) |
| 모델 | SVM only | SVM / XGBoost / RF + **CNN 비교** |
| 해석 | 없음 | SHAP + Feature Importance + Ablation |
| 클래스 수 | 9개 (Normal, Spiccato, Flageolet 등) | 5개 |

**판단:** 같은 데이터셋을 그대로 쓴 "단순 재현(Level 1-2)"은 아닙니다. "동일한 연구 문제를, 다른 데이터셋과 더 엄격한 검증 방식(LOSO)으로 수행하고, 모델 비교와 해석을 추가한 **확장 연구(Level 3-4)**"에 해당합니다. MIR 심사자 입장에서는 "기존 연구의 확장(Extension)"으로 받아들여질 가능성이 가장 높고, 데이터셋·검증 방식이 다르다는 점에서 "독립적 연구"로 볼 여지도 있습니다. "단순 재현"으로 보일 가능성은 낮습니다.

### 2.2 학술적으로 의미 있는 결과

1. **LOSO 환경에서 SVM(0.916) > CNN(0.905), 분산도 SVM이 더 작음(±0.011 vs ±0.030)**
   데이터가 제한적인(3,402개) MIR 분류 과제에서, 수동 특징 + 전통 ML이 mel-spectrogram + CNN보다 더 안정적이라는 실증 사례입니다. "딥러닝이 항상 더 낫다"는 통념에 대한 반례로서 강조할 가치가 있습니다.

2. **Chroma negative control / ablation**
   "Chroma는 기법 분류에 무효일 것"이라는 가설을 실제로 빼보고 검증(변화폭 모두 <1%). 이런 명시적 negative result는 cherry-picking이 아님을 보여주는 좋은 신호입니다.

3. **SHAP 기반 음향학적 해석**
   기법별 결정적 특징을 음향 메커니즘과 연결(예: col_legno_battuto → Spectral Flux 표준편차 최댓값 → 불규칙한 활대 타격). "분류기를 돌렸다"가 아니라 "왜 그렇게 분류되는가"를 설명한 부분이 본 연구의 기여입니다.

### 2.3 보완이 필요한 부분

| 항목 | 현재 상태 | 보완 방향 |
|---|---|---|
| Fold 수 | 4 (악기별 LOSO) | n=4로는 통계적 유의성 검정(t-test 등)이 무리. per-fold 수치를 전부 보고하고 "유의성 검정"이 아닌 "일관성 관찰"로 프레이밍 |
| Related Work | 설계 문서에 산재, 종합된 선행연구 절 없음 | 논문화 시 별도 "Related Work" 절 필요 (3장 참고) |
| CNN 베이스라인 | SmallCNN(~170K params), 사전학습 없음 | PANNs/AST 등 사전학습 모델과 비교하면 "CNN이 SVM보다 못한 이유가 데이터 부족인지 구조 한계인지" 분리 가능 (`04_v2_cnn.md`에서 이미 후속 과제로 언급) |
| 데이터셋 출처 | OrchideaSOL 단일 | 일반화 주장은 OrchideaSOL/SOL 계열로 범위를 한정하는 것이 안전 |
| 언어 | 전체 한국어 | 국제 학회 투고 시 영문화 필요 |

### 2.4 투고처 제안 (난이도 순)

| 투고처 | 형식 | 현재 상태로 가능 여부 |
|---|---|---|
| 국내 학회 (한국음향학회, 한국컴퓨터종합학술대회 학생논문 세션 등) | Short/Full paper | **현재 상태로 충분** |
| ISMIR Late-Breaking/Demo (LBD) | 2p extended abstract | **현재 상태로 가능** — 부담 적고 피드백 받기 좋은 첫 단계 |
| DLfM (Digital Libraries for Musicology, ISMIR 위성 워크숍 — 확장 주법 주제 특화) | Workshop paper | 2.3의 Related Work 보강 후 가능 |
| ISMIR/DAFx full paper, JNMR (Kruger & Jacobs 게재지) | Full paper / Journal | 2.3 전체 보완 + 영문화 필요 |

---

## 3. 아이디어의 출처 — 관련 연구

### 3.1 직접적 기반 (이미 본문에서 인용 중)

- **Cella, C.E., Ghisi, D., Lostanlen, V., Lévy, F., Fineberg, J., Maresz, Y. (2020).** *OrchideaSOL: A Dataset of Extended Instrumental Techniques for Computer-Aided Orchestration.* arXiv:2007.00763.
  → 데이터셋의 출처. 52종 연주기법 라벨이 존재한다는 점에서 본 프로젝트의 5-클래스 설계가 가능해짐.

- **Kruger, A.B. & Jacobs, J.P. (2020).** *Playing Technique Classification for Bowed String Instruments from Raw Audio.* Journal of New Music Research, 49(4), 320–333.
  → "현악기 보잉 아티큘레이션 분류"라는 연구 문제 자체의 출발점. 이 논문의 random 70/30 분할이 가진 데이터 누수 위험에 대한 대응으로 본 프로젝트의 LOSO GroupKFold 설계가 나왔음 (`06_v1_results.md`, `07_v2_results.md` 참고).

### 3.2 방법론적 기반

- **Lundberg, S.M. & Lee, S-I. (2017).** *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
  → SHAP의 원 논문. v1의 해석 단계(`03_v1_modeling.md`, `06_v1_results.md`) 전체가 이 프레임워크 위에 있음.

- **Han, Y., Kim, J., Lee, K. (2017).** *Deep Convolutional Neural Networks for Predominant Instrument Recognition in Polyphonic Music.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 25(1).
  → mel-spectrogram을 이미지처럼 다루는 CNN 분류라는 v2 접근의 일반적 배경 (IRMAS 데이터셋 기반 악기 인식이 원 주제).

### 3.3 추가로 확인해볼 만한 연구 (서지정보 확인 필요)

아래는 주제상 가장 가깝지만 정확한 서지정보(연도·게재처)는 기억에 의존한 것이라 **Google Scholar 등에서 직접 확인 후 인용 여부를 결정하시기를 권장**합니다.

- Lostanlen 등의 "pitch spiral"/scattering transform 기반 악기 인식 연구 (ISMIR 2016 전후) — 음높이에 불변한 표현으로 음색·주법을 분류. OrchideaSOL의 모태인 SOL 데이터셋군과 직접 연결됨.
- Lostanlen 등의 "extended playing techniques" 관련 연구 (DLfM, ISMIR 위성 워크숍) — "확장 주법 인식"을 MIR의 차세대 과제로 제시. 본 프로젝트의 동기(평주법이 아닌 확장 기법 분류)와 가장 직접적으로 맞닿아 있음.

이 두 편이 실제로 확인되면, "OrchideaSOL/SOL 계열에서 확장 주법을 다룬 선행 연구"로서 Related Work의 핵심 축이 될 수 있습니다.

---

## 요약

- **현실성:** 국내 학회 발표나 ISMIR LBD(2p) 수준에는 현재 상태로 충분. Full paper급으로 가려면 Related Work 정리, fold 수 한계에 대한 프레이밍, 사전학습 CNN 비교가 필요.
- **독창성 수준:** "단순 재현"이 아닌 — 다른 데이터셋 + 더 엄격한 검증(LOSO) + 모델 비교 + 해석(SHAP)으로 구성된 **확장 연구(Level 3-4)**.
- **핵심 아이디어 출처:** OrchideaSOL(데이터셋) + Kruger & Jacobs 2020(연구 문제)의 결합에, 더 엄격한 검증·해석·CNN 비교를 더한 것이 본 프로젝트의 본질.
