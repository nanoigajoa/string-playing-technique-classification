# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 상태

현악기 연주기법(ordinario / pizzicato_secco / tremolo / sul_ponticello / col_legno_battuto) 자동 분류 MIR 포트폴리오 프로젝트.

**완료된 작업:**
- **v1:** 245차원 특징 공학 → SVM(F1=0.916) / XGBoost / RF + SHAP 해석
- **v2:** mel-spectrogram → SmallCNN(F1=0.905)

**진행 중 (연구과정/02_계획_확장Phase1-4.md):**
- **Phase 1:** MERT-v1-95M 백본 + Self-Training 파이프라인 (+ CLAP 제로샷 기준선)
- **Phase 2:** Qwen2-Audio-7B-Instruct 환각(hallucination) 평가
- **Phase 3:** SHAP 기반 해석가능성

## 환경 제약

- 하드웨어: MacBook Air M5, 10코어 CPU, 24GB unified memory, GPU 없음
- PyTorch: MPS 백엔드 우선 (`mps` → `cpu` fallback), CUDA 없음
- **유료 API 금지** (OpenAI 등) — 모든 모델은 HuggingFace에서 무료 로드
- **bitsandbytes 사용 금지** — CUDA 전용, MPS 불가. 양자화는 optimum-quanto int8 사용

## 커맨드

```bash
# 의존성 설치 (반드시 venv 안에서)
pip install -r requirements.txt

# v1 학습 + SHAP (노트북)
jupyter notebook notebooks/02_modeling_v1.ipynb

# v2 CNN 학습 (노트북)
jupyter notebook notebooks/03_modeling_v2.ipynb

# 추론 — SVM (기본값)
python src/predict.py path/to/audio.wav

# 추론 — CNN
python src/predict.py path/to/audio.wav --model cnn

# Phase 1/2/3 (독립 실행 스크립트)
python src/phase1_selftraining.py
python src/phase2_alm_eval.py
python src/phase3_shap.py
```

## 코드 구조

```
src/
  features.py        — 245차원 특징 추출 (MFCC+Δ+ΔΔ, Flux, ZCR, RMS, Chroma, LAT)
  dataset.py         — MelDataset + audio_to_mel() + SpecAugment
  model.py           — SmallCNN (비대칭 풀링 (2,1): 주파수축만 절반 감소)
  build_dataset.py   — 파일 목록 구성
  predict.py         — 추론 파이프라인 (svm / cnn 선택)
  phase1_selftraining.py   — [신규] MERT + pseudo-labeling 루프
  phase2_alm_eval.py       — [신규] Qwen2-Audio-7B-Instruct 환각 평가
  phase3_shap.py           — [신규] SHAP ↔ acoustic descriptor 시각화
notebooks/
  01_eda.ipynb
  02_modeling_v1.ipynb     — v1 학습 + 모델 저장 (models/svm_v1.joblib)
  03_modeling_v2.ipynb     — v2 학습 + 모델 저장 (models/cnn_v2.pt)
results/
  v1/  v2/  comparison/    — confusion matrix, SHAP plot, ablation (png + md)
data/
  features_full.csv        — 사전 추출 3,402×245 특징 행렬
_OrchideaSOL2020_release/  — 원본 오디오 (별도 다운로드 필요)
```

## 핵심 설계 원칙

### 데이터 분할 — 반드시 악기 단위 GroupKFold

같은 녹음의 다른 음이 train/test에 동시에 들어가면 정확도 과대평가. **LOSO(Leave-One-Instrument-Out)**을 기준 평가로 사용. 직접 분할 시:

```python
from sklearn.model_selection import GroupKFold
# groups = features.py의 parse_recording_id() 결과
```

TinySOL은 사용 불가 — ordinario만 포함되어 있어 기법 분류 불가.

### 특징 추출: 프레임 → [mean, std, Q25, Q75] 집계

평균만 쓰면 시간적 변화 정보 손실. `src/features.py`의 `_aggregate()`가 4통계를 적용.

**특징 변별력 순서:** Spectral Flux / ZCR / Log-Attack-Time > MFCC+Δ+ΔΔ > Spectral Contrast > Chroma

Chroma는 기법 분류에 무효 (pitch class 정보). 제거하지 말고 ablation 대조군으로 유지.

### Phase 1 — Self-Training

백본 `m-a-p/MERT-v1-95M`을 HuggingFace transformers로 로드. MPS 디바이스 우선, 실패 시 CPU fallback. 매 iteration마다 confusion matrix 저장.

**CLAP 제로샷 기준선 (단계 1-0):** `laion/clap-htsat-unfused`로 텍스트 프롬프트 기반 제로샷 분류를 먼저 측정 (학습 없음, ~300MB). 이 수치가 "라벨 0개 성능" 기준점이 되며 self-training 효과를 대조하는 데 사용된다.

### Phase 2 — ALM 환각 평가

`Qwen2-Audio-7B-Instruct` 로드: `torch_dtype=torch.float16, device_map="auto"` (MPS 라우팅). OOM 시 optimum-quanto int8 fallback. 질문 세트는 코드로 자동 생성 (템플릿 기반, 외부 API 호출 금지).

### Phase 3 — SHAP

Phase 1 head에 shap 적용 후 librosa acoustic descriptor와 SHAP value를 매칭. Phase 2에서 ALM이 가장 많이 틀린 주법에 대해 케이스 스터디 3~5개 작성.

## 평가

- **주 지표:** Macro F1 (클래스 불균형 대응)
- **보조:** Confusion Matrix, Per-class Precision/Recall
- 모든 결과는 `results/` 폴더에 markdown 리포트 + png로 저장

## 모델 역할

| 모델 | 역할 |
|---|---|
| SVM (v1) | 최고 성능 (F1=0.916), 추론 기본값 |
| XGBoost (v1) | SHAP 해석 주 대상 (F1=0.891) |
| SmallCNN (v2) | mel-spectrogram 접근 (F1=0.905) |
| CLAP (laion/clap-htsat-unfused) | Phase 1 제로샷 기준선 (~300MB, 학습 없음) |
| MERT head (Phase 1) | Self-Training 실험 |
| Qwen2-Audio-7B-Instruct (Phase 2) | 환각 평가 대상 (평가만, 학습 없음) |
