# Phase 1 — Self-Training 파이프라인

## 실제 결과 (2026-07-01 누수 수정본)

> ⚠️ 초판(2026-06-30)은 **iteration을 test F1로 골라**(`max(iter_f1)`) 부풀린 누수판이었다.
> "5~50% 전 구간 +7~8%p 향상"과 "단일 fold 0.920" 헤드라인은 **철회**. 아래는 수정본
> (iteration은 검증셋으로 선택, 전 수치 4-fold 평균±표준편차).
> 방법론: [../method_leakage_note.md](../method_leakage_note.md) 함정 1.

| seed 비율 | CLAP 기준선 | MERT baseline | + self-training (τ=0.7) |
|---|---|---|---|
| 5% | 0.363 | 0.726 | 0.638 |
| 10% | 0.363 | 0.725 | 0.739 |
| 20% | 0.363 | 0.777 | 0.798 |
| 50% | 0.363 | 0.821 | 0.829 |

- Self-training 이득 미미(10~50% +0.01~0.02), **5%는 오히려 하락** — 저라벨에서 반복 선택 불안정.
- 유의성: seed 20% +0.021±0.018, Wilcoxon p=0.125 → fold n=4 한계로 확정 불가.
- **공정 튜닝 시 MERT ≈ SVM** (단일 linear head가 MERT 언더튜닝 — 함정 2). 상세 [../method_leakage_note.md](../method_leakage_note.md).
- 결과 파일: `results/phase1/` (report.md, significance.md, svm_seed_comparison.md, mert_tuning_audit.md, learning_curve.png)
- 스크립트: `src/phase1_selftraining.py`, `phase1_rerun.py`, `phase1_significance.py`, `phase1_mert_tuning_audit.py`

---

## 목적

"라벨 부족이 세부 음악 태스크 확장의 실질적 병목"임을 정량적으로 보여준다.
self-training이 잘 작동했다는 증명이 목적이 아니라,
라벨 비율에 따라 성능이 어떻게 변하고 어디서 plateau가 오는지를 드러내는 것이 목적이다.

---

## 왜 MERT인가

v1은 수동으로 설계한 245차원 특징 벡터(MFCC, Spectral Flux 등)를 사용했다.
MERT(Music Encoder Representations from Transformers)는 240k 시간 분량의 음악 데이터로
사전학습된 foundation model이다. 수동 특징 설계 없이도 오디오에서 풍부한 표현을 추출한다.

이 비교가 Phase 1의 부산물이기도 하다:
- **MERT가 v1 SVM(F1=0.916)을 넘으면**: foundation model의 표현 능력이 소규모 전문 데이터에서도 유효함을 보임
- **MERT가 v1을 못 넘으면**: "3,400개 규모에서는 도메인 수동 특징이 범용 foundation model보다 낫다"는 것 자체가 발견 — 실패로 처리하지 말 것

결과가 어느 쪽이든 해석 가능하도록 실험을 설계한다.

---

## CLAP 제로샷 기준선 (단계 1-0)

### 목적

Self-training 실험에 앞서 **"라벨이 전혀 없을 때의 성능"** 기준점을 설정한다.
CLAP(Contrastive Language-Audio Pretraining)은 오디오-텍스트 쌍으로 학습된 모델로,
텍스트 설명만으로 오디오 클래스를 구분한다. 학습 없음, 추론만 실행.

이 수치가 낮을수록 "전문가 라벨의 필요성"을 더 강하게 뒷받침한다. 결과가 어떻게 나오든
해석 가능하다 — CLAP이 낮으면 라벨 필요성 입증, 높으면 텍스트-오디오 정렬 품질 검증.

### 모델 사양

- **모델 ID**: `laion/clap-htsat-unfused`
- **크기**: ~300MB, M5 float32 추론 문제 없음
- **입력 SR**: 48kHz (librosa로 리샘플링)
- **transformers** 패키지에서 `ClapModel` / `ClapProcessor`로 로드 — 추가 패키지 불필요

### 텍스트 프롬프트 설계

클래스별 2종 프롬프트를 실험해 텍스트 표현 민감도를 함께 확인한다.

| 클래스 | 프롬프트 (generic) |
|---|---|
| ordinario | "a string instrument playing with normal arco bowing" |
| pizzicato_secco | "a string instrument playing pizzicato, plucking the string" |
| tremolo | "a string instrument playing tremolo, rapid repeated bow strokes" |
| sul_ponticello | "a string instrument playing sul ponticello, bowing near the bridge" |
| col_legno_battuto | "a string instrument playing col legno battuto, striking with the wood of the bow" |

generic 버전 외에 악기명 명시 버전("a violin playing…")도 비교한다.

### 추론 흐름

```
오디오 (48kHz) → ClapProcessor → ClapModel.get_audio_features() → L2 정규화 (D,)
텍스트 ×5     → ClapProcessor → ClapModel.get_text_features()  → L2 정규화 (5, D)
                                           ↓ cosine similarity → (N, 5)
                                           ↓ argmax → predicted class
```

### 출력

- `results/phase1/clap_zeroshot.csv`: 샘플별 예측 결과
- `results/phase1/clap_confusion.png`: CLAP 제로샷 confusion matrix
- CLAP macro F1 → 학습 곡선 그래프에 수평 기준선으로 추가 (v1 SVM 0.916과 함께)

---

## 데이터 분할 설계 (단계 1-1)

### 핵심 원칙: 악기 단위 완전 분리

v1/v2에서 확립한 원칙을 그대로 유지한다.
`parse_recording_id()`가 반환하는 악기+음+강약 단위를 GroupKFold 그룹 키로 사용.

### 라벨 은닉 방식

외부 비라벨 데이터를 사용하지 않고 OrchideaSOL 자체의 라벨을 일부 숨겨서
비라벨 풀을 시뮬레이션한다. 이 방식을 선택한 이유:

- **oracle 보유**: 숨긴 라벨로 pseudo-label 정확도를 직접 측정 가능
- **도메인 일치**: 외부 데이터의 도메인 shift가 실험 변수로 개입하지 않음
- **학습 곡선 실험**: seed 비율을 단계적으로 바꾸면서 "라벨 몇 %부터 self-training이 의미 있는가"를 측정 가능

이 설정은 semi-supervised learning 논문에서 표준적으로 사용하는 "partially-labeled" 프로토콜이다.

### Seed 비율 설계

| Labeled seed | 샘플 수 (근사) | 의미 |
|---|---|---|
| 5% | ~170개 | 거의 라벨 없는 극단 케이스 |
| 10% | ~340개 | 초소형 라벨 세트 |
| 20% | ~680개 | 소형 라벨 세트 |
| 50% | ~1,700개 | 절반 라벨 — self-training 효과가 사라지는 지점 예상 |

각 비율마다 random seed를 달리해 **3 trial** 반복.
클래스 불균형이 있으므로 반드시 stratified sampling.

### 분할 순서

1. GroupKFold 그룹 키 생성 (parse_recording_id 결과)
2. 전체 데이터를 그룹 단위로 shuffling
3. 각 trial에서 stratified sampling으로 labeled seed 추출
4. 나머지 전체 = unlabeled pool (oracle 라벨 보유, 추론 시에만 숨김)

분할 설정은 `data/split_configs.json`에 저장해 재현 가능성 보장.

---

## MERT 임베딩 추출 (단계 1-2)

### 모델 사양

- **모델 ID**: `m-a-p/MERT-v1-95M`
- **파라미터 수**: 95M → float32 약 380MB, 24GB unified memory에서 문제 없음
- **디바이스**: MPS 우선 시도. `torch.backends.mps.is_available()` 확인 후 fallback to CPU
- **출력**: 마지막 hidden layer의 time-axis mean pool → 768차원 벡터

### 추출 전략

MERT는 raw waveform을 입력으로 받는다. 각 오디오 파일을 16kHz로 리샘플링 후
feature extractor를 통과시키면 (1, time_frames, 768) 형태의 hidden states가 나온다.
마지막 레이어의 time dimension을 mean pooling해 고정 길이 벡터로 변환.

```
audio (raw waveform) → MERT feature extractor → hidden states (time, 768)
                                                        ↓ mean pool
                                                  embedding (768,)
```

전체 3,402개 샘플에 대해 **한 번만 실행하고 numpy 파일로 캐싱**.
매 실험마다 MERT forward pass를 반복하면 M5에서 수 시간이 걸린다.

캐시 파일:
- `data/mert_embeddings.npy` — shape: (3402, 768), dtype: float32
- `data/mert_labels.npy` — shape: (3402,), dtype: int, 클래스 인덱스
- `data/mert_filenames.txt` — 행 순서와 파일명 대응

### 주의사항

MERT는 기본적으로 24kHz 또는 16kHz를 가정한다. OrchideaSOL 원본 샘플링 레이트를 확인하고
feature extractor의 `sampling_rate` 파라미터와 일치시켜야 한다.
`librosa.load(filepath, sr=16000)` 또는 MERT processor에 sr을 명시.

---

## Classification Head 설계 (단계 1-3)

### Head 구조

MERT 백본은 학습 중 **동결(frozen)**. Head만 학습한다.
이유:
- 24GB M5에서 95M 파라미터 전체를 학습하면 메모리와 시간이 과도하게 소요
- "MERT 표현이 이 태스크에 얼마나 유용한가"를 측정하려면 백본 변경 없이 head만 평가하는 것이 더 깨끗한 실험 설계

Head 구조 옵션:
- **Linear (권장)**: 768 → n_classes (5), 파라미터 최소화, 해석 용이
- **MLP**: 768 → 256 → n_classes, 표현력 증가 — Linear가 낮으면 시도

### 학습 설정

- Loss: CrossEntropyLoss with class weight (클래스 불균형 대응, v1과 동일 방식)
- Optimizer: AdamW, lr=1e-3
- Epochs: 최대 100, early stopping (patience=10, val macro F1 기준)
- Validation: labeled seed의 20%를 validation으로 분리

### 기준선 (Baseline) 측정

self-training 루프 시작 전, labeled seed만으로 학습한 결과를 먼저 기록한다.
이것이 "라벨 X%만 있을 때 순수 지도학습 성능"의 기준선이 된다.

| 기록 항목 | 용도 |
|---|---|
| Macro F1 (labeled seed 비율별) | 학습 곡선 x축과 y축 |
| Per-class F1 | 어떤 클래스가 라벨 부족에 민감한가 |
| Confusion matrix | Phase 3 케이스 스터디 대상 클래스 예비 선정 |

---

## Self-Training 루프 (단계 1-4)

### 알고리즘 개요

```
1. labeled seed로 head 학습 (baseline)
2. unlabeled pool에 대해 예측 → softmax 확률 계산
3. max probability > τ 인 샘플을 pseudo-label과 함께 labeled set에 추가
4. 확장된 labeled set으로 head 재학습
5. 2~4 반복 (최대 5 iteration, 또는 조기 종료 조건 충족 시)
```

### Confidence Threshold τ

τ를 {0.7, 0.8, 0.9}로 sweep한다.

| τ 값 | 특성 |
|---|---|
| 0.7 | 채택률 높음, 오염 위험 높음 |
| 0.8 | 균형점 (일반적으로 자주 사용) |
| 0.9 | 채택률 낮음, 정확도 높음, 학습 효과 약함 |

τ 선택이 성능에 미치는 영향이 Phase 1의 주요 결과 중 하나다.

### 매 Iteration에서 기록할 항목

| 항목 | 의미 |
|---|---|
| Macro F1 (val 기준) | 성능 변화 추이 |
| 채택률 (accepted / total unlabeled) | self-training이 얼마나 데이터를 흡수하는가 |
| 클래스별 채택 수 | 특정 클래스 쏠림 현상 발생 여부 |
| Pseudo-label 정확도 | oracle 라벨로 실제 정확도 확인 (이것이 라벨 은닉 방식의 핵심 장점) |
| Confusion matrix PNG | `results/phase1/seed{ratio}_trial{n}_tau{t}_iter{i}.png` |

### 조기 종료 조건

다음 중 하나 충족 시 loop 종료:
1. 채택률 < 1% (추가할 고확신 샘플이 거의 없음)
2. Val macro F1이 이전 iteration 대비 0.005 이상 개선되지 않음 (3 iteration 연속)
3. 최대 5 iteration 도달

### 클래스 불균형 주의사항

unlabeled pool에서 multi수 클래스(ordinario, tremolo)가 pseudo-label을 독점할 수 있다.
매 iteration마다 클래스별 채택 수를 기록해두고,
최종 분석에서 "pseudo-labeling이 특정 클래스를 과대 채택했는가"를 확인한다.
필요 시 클래스별 채택 상한(max 채택 수 = labeled seed의 클래스 분포 기준)을 도입할 수 있다.

---

## 결과 시각화 (단계 1-5)

### 학습 곡선 (핵심 그래프)

x축: labeled seed 비율 (5%, 10%, 20%, 50%)
y축: Macro F1

두 선을 같은 그래프에:
- **Baseline**: 지도학습만 (self-training 없음)
- **Best self-training**: 해당 seed 비율에서 τ 3종 중 최고 F1

이 그래프에서 "self-training이 도움이 되는 구간"과 "baseline과 차이가 없는 구간"이 시각적으로 드러난다.

### τ별 트레이드오프 곡선

x축: τ 값 (0.7, 0.8, 0.9)
y축 좌: Final macro F1
y축 우: 최종 채택률 (%)

"더 엄격한 threshold가 항상 좋은가, 아니면 적당한 오염을 감수하고 더 많은 데이터를 흡수하는 것이 낫는가"를 보여주는 그래프.

### 리포트 구성

`results/phase1/report.md`:
1. 실험 설정 요약 (seed 비율, τ, trial 수)
2. 학습 곡선 이미지 삽입
3. "self-training이 가장 효과적인 조건" 서술
4. "한계가 나타나는 지점" 서술 — 이것이 Phase 2/3 연결 지점
5. v1 SVM(0.916)과의 비교 — "MERT head가 전체 라벨을 써도 SVM을 넘지 못하면" 그 이유 해석

---

## v1 SVM과의 비교 방법

MERT head(full label)와 v1 SVM을 공정하게 비교하려면 동일한 평가 프로토콜이 필요하다.

v1은 GroupKFold n=4, LOSO 방식이었다.
Phase 1 seed 100%(= 전체 라벨)에서 동일한 GroupKFold n=4로 평가하면 직접 비교 가능.

이 비교를 통해:
- "MERT가 SVM 이상이면" → foundation model 표현이 유효
- "MERT가 SVM 미만이면" → "소규모 전문 데이터에서 수동 특징이 여전히 경쟁력 있다"는 v1 결과를 재확인하는 별도 발견

---

## 주요 위험 요소

| 위험 | 대응 |
|---|---|
| MERT MPS 로드 실패 | CPU fallback 자동 전환, 임베딩 추출은 속도가 느려도 1회만 하므로 허용 |
| 임베딩 추출 시간 과다 | 3,402개 × 1회 추출 후 캐싱. 반복 실험은 캐시 사용 |
| Pseudo-label 오염 누적 | 매 iteration pseudo-label 정확도 기록, 급격한 F1 하락 시 이전 iteration 체크포인트로 rollback |
| 클래스 쏠림 | 클래스별 채택 수 모니터링, 필요 시 클래스별 채택 상한 도입 |
