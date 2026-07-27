# 미세 현악 주법 분류에서의 세 가지 평가 함정

**"파운데이션 모델이 전통적 특징+SVM보다 우월해 *보이는* 결과는 세 가지 평가 함정의 산물이며,
함정을 제거하면 두 모델은 통계적으로 동급이다."**

현악기 5가지 주법(ordinario / pizzicato_secco / tremolo / sul_ponticello / col_legno_battuto)
분류를 소재로, **오디오 AI 모델을 어떻게 공정하게 평가하는가**를 다루는 MIR 연구. 성능 경쟁이
아니라, 실험 과정에서 실제로 마주친 세 가지 평가 함정을 전후 비교로 정량화하고, 왜 무거운 모델이
이 과제에서 실질적 이득을 주지 않는지를 층별 분석으로 설명한다.

> 📄 논문 초안: [`연구과정/주법분류_논문초안_국문_v0.md`](연구과정/주법분류_논문초안_국문_v0.md)
> · 영문: [`주법분류_논문초안_영문_v0.md`](연구과정/주법분류_논문초안_영문_v0.md)

---

## 핵심 서사 — 결론이 두 번 뒤집힌 이야기

| 단계 | 관찰 | 실체 |
|---|---|---|
| 처음 | 제한 라벨에서 MERT(파운데이션 모델)가 SVM보다 우세해 보임 | — |
| **함정 ① (1차 반전)** | 자기학습 반복 회차를 **test F1로 선택** → 선택-시점 누수 | 제거하니 5%에서 −0.107, 교차점 소멸 |
| **함정 ② (2차 반전)** | 마지막 층 단일 선형 헤드 → **백본 언더튜닝** | 관행대로 튜닝하니 MERT +0.025~0.046, **동급** |
| **함정 ③ (미수)** | 클립 단위 부트스트랩 → **재표집 단위 오설정** | fold 단위로 재표집하면 CI가 전 구간 0 포함 |
| **착지** | — | **조건부 동급**: 튜닝 SVM ≈ 튜닝 MERT (±0.7 fold-std, n=4) |

함정 ①②는 실제로 결론을 뒤집었고, ③은 뒤집히기 **직전에 포착**했다.

---

## 주요 결과

### 튜닝 예산 스펙트럼 (macro F1, LOSO, n=4)

| seed | SVM(튜닝) | frozen-linear | frozen-tuned | layer-weighted | LoRA |
|---|---|---|---|---|---|
| 20% | 0.832 | 0.776 | 0.807 | **0.855** | 0.804 |
| 50% | 0.868 | 0.814 | 0.859 | **0.900** | 0.775 |

- MERT의 공정한 상한은 **layer-weighted 프로브**로, SVM과 동급이거나 약간 위.
- 백본을 **LoRA로 미세조정하면 오히려 퇴행**한다(데이터가 많을수록 악화) — 분포 외(교차악기)
  평가에서 미세조정이 사전학습 특징을 왜곡하는 **LP-FT 현상**. "덜 튜닝해서 졌다"는 반박을 격파.

### 전체 데이터 기반 모델 (baseline, full-label LOSO)

| Model | Macro F1 | Std | Track |
|---|---|---|---|
| SVM (RBF) | 0.916 | ±0.011 | v1 (수동 특징 245차원) |
| CNN (SmallCNN) | 0.905 | ±0.030 | v2 (멜 스펙트로그램) |
| XGBoost (tuned) | 0.893 | ±0.032 | v1 (SHAP 대리 모델) |

평가: **Leave-One-Instrument-Out** (GroupKFold n=4) — 학습에 쓰지 않은 악기 전체로 테스트.

---

## 왜 깊이가 필요 없는가 (메커니즘)

- **층별 서술자 프로브:** 주법을 가르는 핵심 단서 onset-strength envelope는 이미 MERT의 합성곱
  프론트엔드(L0)에서 R²=0.767로 포화. 12개 트랜스포머 층이 더하는 것은 대역 평균 +0.008뿐.
- **SHAP:** onset-strength가 차원 보정 중요도 1위(0.481), Chroma는 뒤에서 2번째(음성 대조군).
  flux 방향도 물리와 일치(col legno=낮은 변동, tremolo=높은 변동).
- **정직한 음성 결과:** 층↔음악개념 정렬 주장은 지지되지 않아 null로 보고. SHAP↔ALM 인과는 관찰만.

---

## 데이터셋

- **OrchideaSOL** (Ircam Forum) — Violin, Viola, Violoncello, Contrabass
- 총 **3,402개** 샘플 (자연 클래스 불균형)

| 기법 | 설명 | 샘플 수 |
|---|---|---|
| ordinario | 일반 활주법 | 1,193 |
| tremolo | 빠른 반복 보잉 | 1,041 |
| pizzicato_secco | 손가락으로 현을 뜯기 | 639 |
| sul_ponticello | 브릿지 근처에서 활질 | 302 |
| col_legno_battuto | 활대 나무로 타격 | 227 |

---

## 프로젝트 구조

```
├── 연구과정/
│   ├── 주법분류_논문초안_국문_v0.md/.docx   # 논문 초안 (국문, 정독용)
│   ├── 주법분류_논문초안_영문_v0.md/.docx   # 논문 초안 (영문)
│   └── 14~21_*.md                          # 섹션별·통합 초안
├── src/
│   ├── features.py            # 특징 추출 (245차원: MFCC+Δ+ΔΔ, flux, ZCR, RMS, chroma, LAT)
│   ├── dataset.py / model.py  # 멜 스펙트로그램 Dataset + SmallCNN
│   ├── build_dataset.py       # features_full.csv 재생성
│   ├── predict.py             # 추론 (svm / cnn)
│   ├── phase1_*.py            # MERT 감사·튜닝·부트스트랩·LoRA(LP-FT)
│   ├── phase2_alm_eval.py     # Qwen2-Audio 평가
│   └── phase3_*.py, phase4_*  # SHAP·descriptor probe / 교차도메인
├── results/
│   ├── 누수제거_전후비교.md               # 함정 ①
│   ├── phase1/mert_tuning_audit.md       # 함정 ②
│   ├── phase1/cluster_bootstrap.md       # 함정 ③
│   ├── phase1/lora_spectrum.md           # 튜닝예산 스펙트럼 (LP-FT)
│   └── phase3/descriptor_probe.md, report.md  # 메커니즘·SHAP
├── docs/                     # 단계별 설계·분석 문서
├── data/features_full.csv    # 사전 추출 3,402×245 특징 행렬
└── requirements.txt
```

> 대용량 캐시(MERT 임베딩 `.npy`, 원본 오디오, `.zip`)는 재생성 가능하여 저장소에서 제외됨.

---

## 설치

```bash
pip install -r requirements.txt
```

> OrchideaSOL은 [Ircam Forum](https://forum.ircam.fr) 가입 후 별도 다운로드하여
> `_OrchideaSOL2020_release/`에 배치. 환경 제약: Apple Silicon MPS(CPU 폴백), CUDA/bitsandbytes 미사용.

---

## 재현 / 추론

```bash
# 특징 행렬 재생성
python src/build_dataset.py

# 세 함정 감사 (MERT 튜닝·부트스트랩)
python src/phase1_mert_tuning_audit.py
python src/phase1_cluster_bootstrap.py

# 튜닝예산 스펙트럼 (LoRA, --rest 로 냉각 대기 조절)
python src/phase1_lora.py

# 추론 — SVM(기본) / CNN
python src/predict.py path/to/audio.wav
python src/predict.py path/to/audio.wav --model cnn
```

v1/v2 baseline 학습: `notebooks/02_modeling_v1.ipynb`, `03_modeling_v2.ipynb`.

---

## 참고 문헌 (일부)

- Kruger & Jacobs (2020, JNMR); Su, Lin & Yang (2014, TASLP) — 활현악기 주법 분류.
- Oliver et al. (2018, NeurIPS) — 튜닝된 baseline과 현실적 준지도 평가.
- Kapoor & Narayanan (2023, Patterns) — 누수와 재현성 위기.
- Bouthillier et al. (2021, MLSys) — 벤치마크 분산 분해(재표집 단위).
- Pasad et al. (2021, ASRU) — SSL 음성 모델 층별 분석. Kumar et al. (2022, ICLR) — LP-FT.
- Li et al. (2024, ICLR) — MERT. Cella et al. (2020) — OrchideaSOL.
