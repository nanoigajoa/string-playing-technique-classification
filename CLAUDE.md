# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

현악기 연주기법(Arco / Pizzicato / Tremolo / Spiccato 등)을 자동 분류하는 MIR(Music Information Retrieval) 포트폴리오 프로젝트.

- **v1:** 오디오 특징 엔지니어링 → XGBoost / SVM 비교 + Feature Importance 해석
- **v2:** mel-spectrogram → CNN, v1과 트레이드오프 비교

상세 실행 계획: `docs/project-plan.md`

## 예상 기술 스택

- **오디오 처리:** librosa
- **ML 모델:** scikit-learn (Random Forest, SVM), XGBoost, (v2) PyTorch 또는 TensorFlow
- **해석:** SHAP
- **워크플로:** Jupyter Notebook (재현 가능성 최우선)

## 데이터

- **데이터셋:** OrchideaSOL — Ircam Forum 가입 후 다운로드 필요
- **TinySOL은 사용 불가** — ordinario(평주법)만 포함, 기법 분류 불가
- **공식 5-fold 분할** 메타데이터를 그대로 사용 (악기·기법·음역·강약 균형 보장)
- 직접 분할 시 반드시 `GroupKFold(groups=악기_녹음_단위)` 사용 — 같은 녹음의 다른 음이 train/test에 동시에 들어가면 정확도 과대평가됨

## 핵심 설계 원칙

### 특징 추출: 프레임 → 고정 벡터 집계

XGBoost 입력을 위해 가변 길이 오디오를 고정 길이 벡터로 변환해야 한다. 집계 통계는 mean + std + Q25 + Q75 — 평균만 쓰면 시간적 변화 정보가 손실된다.

```python
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    zcr  = librosa.feature.zero_crossing_rate(y)
    feats = []
    for f in [mfcc, np.atleast_2d(flux), zcr]:
        feats += [f.mean(1), f.std(1),
                  np.percentile(f, 25, axis=1),
                  np.percentile(f, 75, axis=1)]
    return np.concatenate(feats)
```

### 특징 우선순위

기법 변별력이 높은 순서: Spectral Flux / ZCR / Log-Attack-Time (과도부 포착) > MFCC+Δ+ΔΔ > Spectral Contrast > **Chroma (음성 대조군)**

Chroma는 기법 분류에 무효 (pitch class 정보라 Pizzicato/Arco를 구분 못 함). 제거하지 말고 "왜 안 되는지 증명하는 대조군"으로 명시적으로 포함할 것.

### 클래스 설계

쉬운 기법(Arco vs Pizzicato)만 쓰면 99% 정확도가 나와 배울 게 없다. 어려운 보잉 아티큘레이션(Spiccato / Détaché / Martelé / Tremolo vs Arco)을 반드시 포함할 것.

## 모델 역할

| 모델 | 역할 |
|---|---|
| Random Forest | 베이스라인 |
| XGBoost | 주력 모델 (Feature Importance / SHAP 해석 대상) |
| SVM | 비교 모델 |
| CNN (v2) | mel-spectrogram 기반, v1과 트레이드오프 비교 |

## 평가

- **주 지표:** Macro F1 (클래스 불균형 대응)
- **보조:** Confusion Matrix, Per-class Precision/Recall
- v1/v2 비교 시 동일한 평가 프로토콜 사용
