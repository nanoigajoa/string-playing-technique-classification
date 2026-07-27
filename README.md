# 현악기 연주기법 분류 (String Playing Technique Classifier)

OrchideaSOL 데이터셋을 사용해 현악기 5가지 연주기법을 자동 분류하는 MIR 포트폴리오 프로젝트.

**v1:** 수동 특징 추출 → XGBoost / SVM / Random Forest + SHAP 해석  
**v2:** 멜 스펙트로그램 → CNN (SmallCNN with asymmetric pooling)

---

## 결과 요약

| Model | Macro F1 | Std | Track |
|---|---|---|---|
| **SVM (RBF)** | **0.916** | ±0.011 | v1 |
| CNN (SmallCNN) | 0.905 | ±0.030 | v2 |
| XGBoost (tuned) | 0.893 | ±0.032 | v1 |
| Random Forest | 0.886 | ±0.060 | v1 |

평가: **Leave-One-Instrument-Out** (GroupKFold n=4) — 학습에 사용하지 않은 악기 전체로 테스트

---

## 데이터셋

- **OrchideaSOL** (Ircam Forum) — Violin, Viola, Violoncello, Contrabass
- 총 **3,402개** 샘플 (자연 클래스 불균형: ordinario 1,193개 ~ col_legno_battuto 227개)

| 기법 | 설명 | 샘플 수 |
|---|---|---|
| ordinario | 일반 활주법 | 1,193 |
| tremolo | 빠른 반복 보잉 | 1,041 |
| pizzicato_secco | 손가락으로 현을 뜯기 | 639 |
| sul_ponticello | 브릿지 근처에서 활질 | 302 |
| col_legno_battuto | 활대 나무로 타격 | 227 |

---

## 이 프로젝트만의 기여

Kruger & Jacobs (2020, JNMR) — 이 분야 대표 논문 — 과 비교:

| 항목 | Kruger & Jacobs 2020 | 이 프로젝트 |
|---|---|---|
| 데이터셋 | RWC database | OrchideaSOL (다른 데이터셋) |
| 검증 방법 | 70/30 random split × 20회 | **LOSO GroupKFold** (악기 단위 완전 분리) |
| 모델 | SVM only | SVM + XGBoost + RF + **CNN 비교** |
| 해석 | 없음 | **SHAP + Feature Importance + Ablation** |
| 클래스 | Normal/Spiccato/Flageolet 등 9/11기법 (9-class + 11-class) | col_legno_battuto 포함 5개 |

**핵심 차이:** 논문의 random 70/30은 같은 연주자 녹음이 train/test에 동시 등장할 수 있음. 본 프로젝트의 LOSO는 테스트 시 한 번도 본 적 없는 악기 전체로 평가 — 더 엄격하고 현실적인 검증.

---

## 프로젝트 구조

```
├── notebooks/
│   ├── 01_eda.ipynb          # 탐색적 데이터 분석
│   ├── 02_modeling_v1.ipynb  # v1: 특징 공학 + ML
│   └── 03_modeling_v2.ipynb  # v2: 멜 스펙트로그램 + CNN
├── src/
│   ├── features.py           # 특징 추출 (245차원)
│   ├── dataset.py            # PyTorch Dataset + 멜 스펙트로그램 변환
│   ├── model.py              # SmallCNN 정의
│   ├── build_dataset.py      # 파일 목록 구성
│   └── predict.py            # 추론 파이프라인
├── results/
│   ├── v1/                   # v1 결과 (confusion matrix, SHAP, ablation 등)
│   ├── v2/                   # v2 결과 (confusion matrix, mel samples)
│   └── comparison/           # v1 vs v2 최종 비교
├── models/
│   ├── svm_v1.joblib         # 저장된 SVM 파이프라인
│   └── cnn_v2.pt             # 저장된 CNN 가중치
├── docs/                     # 단계별 설계 문서 및 결과 분석
├── data/
│   └── features_full.csv     # 사전 추출된 3,402×245 특징 행렬
└── requirements.txt
```

---

## 설치

```bash
pip install -r requirements.txt
```

> OrchideaSOL 데이터셋은 [Ircam Forum](https://forum.ircam.fr) 가입 후 별도 다운로드 필요.
> 다운로드 후 `_OrchideaSOL2020_release/` 디렉토리에 배치.

---

## 재현 방법

```bash
# 1. v1: 특징 추출 + ML 학습 + SHAP
jupyter notebook notebooks/02_modeling_v1.ipynb

# 2. v2: 멜 스펙트로그램 + CNN 학습
jupyter notebook notebooks/03_modeling_v2.ipynb
```

각 노트북 끝에 모델 저장 셀이 있음 — 실행하면 `models/` 폴더에 저장됨.

---

## 추론 (새 오디오 파일 예측)

모델을 저장한 뒤 사용:

```bash
# SVM (v1) — 기본값, 빠르고 안정적
python src/predict.py path/to/audio.wav

# CNN (v2)
python src/predict.py path/to/audio.wav --model cnn
```

출력 예시:
```
[SVM (v1)] sample.wav
  예측 기법:  pizzicato_secco
  확신도:     94.2%

  클래스별 확률:
    pizzicato_secco        0.942  ████████████████████████████
    ordinario              0.031  █
    col_legno_battuto      0.014
    tremolo                0.008
    sul_ponticello         0.005
```

---

## 주요 발견

- **Onset-strength envelope와 ΔMFCC가 핵심 특징** — 어택 특성(어떻게 소리가 시작되는가)이 기법 변별의 핵심
- **Chroma는 spurious signal** — 피치 클래스 정보는 기법 분류에 무효, 데이터 분포 아티팩트
- **SVM이 CNN보다 낫다** — 3,402개 데이터에서는 마진 최적화가 딥러닝보다 견고
- **col_legno_battuto가 가장 어렵다** — 타악적 특성이 현악기 오디오에서 비정형적

---

## 참고 문헌

- Kruger, A.B. & Jacobs, J.P. (2020). *Playing technique classification for bowed string instruments from raw audio.* Journal of New Music Research, 49(4), 320–333.
- OrchideaSOL: Cella, A. et al. (2020). *OrchideaSOL: A dataset of extended instrumental techniques for computer-aided orchestration.*
