# Phase 4 — Philharmonia 교차 도메인 검증 + Self-Training 확장

## 실제 결과 (2026-07-01 누수 수정본)

> ⚠️ 초판(2026-06-30)은 **τ를 Pool B(holdout)로 골라** 부풀린 누수판이었다("τ=0.8, 0.461, +3.8%p" — 철회).
> 아래는 **τ를 OrchideaSOL 소스 val에서 선택**(타깃 정보 미사용)하고 Pool B는 최종 1회만 평가한 수정본.
> 방법론: [../method_leakage_note.md](../method_leakage_note.md) 함정 1.

**데이터**: Philharmonia Strings 577개 (ordinario 300 상한) → Pool A 403개 / Pool B 174개

| 조건 | Pool B Macro F1 |
|---|---|
| in-domain 기준 (OrchideaSOL **LOSO**) | 0.8925 |
| Zero-shot 기준선 (OrchideaSOL → Philharmonia) | 0.4226 |
| **Self-training (τ=0.7, 소스 val 선택)** | **0.4492** |
| 향상폭 | +0.0266 |

> **도메인 이동 손실 = LOSO 0.8925 − zero-shot 0.4226 = −0.47.** in-domain 기준은
> in-sample 재예측(0.9951, 낙관 착시)이 아니라 보지 못한 악기로의 일반화(LOSO)로 잡는다.
> self-training으로 0.4492까지 일부 회복하지만 손실의 대부분은 남는다 — cross-domain은 어렵다.

**Per-class (self-training τ=0.7):**

| 클래스 | Self-training F1 |
|---|---|
| ordinario | 0.847 |
| pizzicato_secco | 0.626 |
| tremolo | 0.773 |
| **sul_ponticello** | **0.000** |
| **col_legno_battuto** | **0.000** |

**핵심 발견:**
- τ 선택이 소스 val에서 세 값 모두 동일(0.9634)로 나옴 — 소스 성능 포화로 τ 선택 신호가 약함(한계).
- Pseudo-label 정확도 ~68% (iter1 0.686 / iter2 0.675) — 독립 도메인에서도 절반 이상 맞는 신호 생성.
- col_legno_battuto / sul_ponticello: self-training 후에도 F1=0.000 — **도메인 이동에 구조적 취약** (이 발견은 유지).
- Phase 3 연결: Spectral Flux 방향성이 이 두 클래스에서 도메인 간 유지되지 않음을 시사.
- 결과 파일: `results/phase4/` (report.md, poolB_*_confusion.png, pseudolabel_accuracy_by_class.png)
- 스크립트: `src/phase4_crossdomain.py`(`_select_tau_on_source`), `data/build_philharmonia_splits.py`

---

## 목적

Phase 1 self-training의 구조적 약점을 보완하고, MERT 기반 분류기의 도메인 일반화 능력을 실증한다.

Phase 1은 OrchideaSOL 라벨을 인위적으로 가려서 비라벨 데이터를 시뮬레이션했다.
이는 "같은 데이터에서 라벨만 숨긴 것"이라는 지적에 취약하다.
Philharmonia는 완전히 다른 출처의 독립 녹음이면서 정답 라벨도 갖고 있다.
이것을 "라벨을 아는 비라벨 데이터"로 사용하면 두 가지를 동시에 달성한다:

1. **진짜 cross-domain self-training 검증** — 독립 데이터에서 pseudo-labeling이 유효한지
2. **Pseudo-label 품질 직접 측정** — 보통 self-training 연구에서는 불가능한 검증. 우리는 답을 알고 있다

### Phase 1과의 관계

Phase 4는 Phase 1의 확장이지 독립 실험이 아니다.
동일한 MERT 백본과 Linear head 구조를 그대로 사용하며, 실험 설정도 Phase 1을 따른다.

```
Phase 1: OrchideaSOL(3,402개) → MERT self-training → 같은 도메인 내 라벨 효율 측정
Phase 4: OrchideaSOL 학습 모델 → Philharmonia(독립 도메인) → cross-domain self-training
```

---

## 왜 단순 병합 재학습을 안 하는가

단순히 Philharmonia를 OrchideaSOL에 합쳐서 재학습하면:

- ordinario가 3,072개로 극단적 불균형 (sul_ponticello 42개의 73배)
- MP3/WAV 코덱 차이가 실험 변수로 개입
- "성능 향상이 self-training 덕인지 데이터가 늘어서인지" 분리 불가
- "독립된 두 도메인" 비교라는 실험의 깨끗함이 사라짐

---

## 데이터 현황

**위치**: `data/philharmonia/Strings/{violin,viola,cello,double bass}/`

**파일명 형식**: `{instrument}_{pitch}_{duration}_{dynamic}_{technique}.mp3`

**클래스 매핑**:

| 우리 클래스 | Philharmonia 파일명 패턴 | 샘플 수 |
|---|---|---|
| ordinario | `*arco-normal*` | 3,072 → **300으로 상한** |
| pizzicato_secco | `*pizz-normal*` | 105 |
| col_legno_battuto | `*arco-col-legno-battuto*` | 73 |
| tremolo | `*arco-tremolo*` | 57 |
| sul_ponticello | `*arco-sul-ponticello*` | 42 |

ordinario 상한을 300으로 두는 이유: 무제한으로 들어오면 기법 클래스 신호가 희석되고,
"기법 분류에서의 self-training 효과" 측정이라는 목적이 묻힌다.

- `arco-col-legno-tratto`는 제외 (battuto와 다른 기법)
- sul_ponticello: 바이올린 39개 집중, 나머지 악기 각 1개
- 포맷: MP3 → `librosa.load(path, sr=16000)` (MERT 입력 sr 통일)

---

## Pool A / Pool B 분할 (단계 4-1)

### 분할 원칙

전체 Philharmonia 샘플을 클래스 비율 유지(stratified) 70/30으로 분리.

```
Pool A (70%) — self-training용, 라벨을 "안 보이는 척"
Pool B (30%) — 평가 전용, 어떤 학습에도 절대 사용 금지
```

col_legno_battuto 73개가 최솟값이므로 stratified split 필수.
Pool B에 최소 22개(73 × 0.3)가 확보되어야 신뢰 가능한 per-class 평가가 된다.

### 코드 구조

```python
# data/build_philharmonia_splits.py

TECHNIQUE_MAP = {
    'arco-normal':            'ordinario',
    'pizz-normal':            'pizzicato_secco',
    'arco-col-legno-battuto': 'col_legno_battuto',
    'arco-tremolo':           'tremolo',
    'arco-sul-ponticello':    'sul_ponticello',
}
ORDINARIO_CAP = 300

# 파일명 파싱 → (path, label) 목록 생성
# stratified split 70/30
# data/philharmonia_pool_a.json, data/philharmonia_pool_b.json 저장
```

---

## Philharmonia MERT 임베딩 추출 (단계 4-2)

Phase 1에서 OrchideaSOL에 대해 했던 작업과 동일하게 Philharmonia에도 적용한다.

- **모델 ID**: `m-a-p/MERT-v1-95M` (Phase 1과 동일)
- **입력 sr**: 16kHz (Phase 1과 동일)
- **출력**: 마지막 hidden layer time-axis mean pool → 768차원 벡터

```
Philharmonia MP3 → librosa.load(sr=16000) → MERT feature extractor
                                                     → hidden states (time, 768)
                                                     → mean pool → embedding (768,)
```

**캐시 파일:**
- `data/philharmonia_mert_embeddings.npy` — Pool A + Pool B 전체, shape: (N, 768)
- `data/philharmonia_mert_labels.npy` — 대응 라벨 벡터
- `data/philharmonia_mert_meta.csv` — 파일 경로, 클래스, pool 구분(A/B)

Phase 1 캐시(`data/mert_embeddings.npy`)와 독립 파일로 관리.
임베딩 추출은 1회만 실행하고 이후 실험은 캐시를 사용한다.

---

## Zero-Shot 교차 도메인 평가 (단계 4-3)

OrchideaSOL로 학습된 모델들을 Pool B에 그대로 적용한다. Fine-tuning 없음.

| 모델 | 학습 기반 | Pool B 평가 방식 |
|---|---|---|
| **MERT head** (주) | OrchideaSOL MERT 임베딩 | `data/philharmonia_mert_embeddings.npy`의 Pool B 부분으로 predict |
| SVM v1 (참고) | OrchideaSOL features.py | `src/features.py`로 특징 추출 후 predict |
| CNN v2 (참고) | OrchideaSOL mel-spec | `src/dataset.py`의 `audio_to_mel()` 후 predict |

MERT가 주 비교 대상 (Phase 1과 같은 모델), SVM/CNN은 "어떤 모델이 cross-domain에서 더 강인한가"를 보는 참고 비교.

**기록:**
- Pool B Macro F1 (cross-domain baseline)
- Per-class F1 — Phase 1 OrchideaSOL 내부 결과와 비교해 도메인 이동 손실폭 측정

---

## Pseudo-Labeling + Self-Training (단계 4-4)

### 흐름

```
Phase 1에서 학습된 MERT head (OrchideaSOL 기반)
        │
        ▼
  Pool A 전체 MERT 임베딩으로 예측 → softmax 확률
        │
        ▼ confidence threshold τ (Phase 1과 동일: 0.7 / 0.8 / 0.9 sweep)
  τ 초과 샘플 → pseudo-label 채택
        │
        ▼
  OrchideaSOL MERT 임베딩 + pseudo-labeled Pool A MERT 임베딩으로 head 재학습
  (백본 MERT는 동결 유지 — Phase 1과 동일 원칙)
```

### Pseudo-Label 정확도 검증

```python
# pseudo_labels: 모델이 붙인 라벨 (τ 초과 샘플)
# true_labels:   Philharmonia 파일명에서 파싱한 실제 라벨
pseudo_accuracy = (pseudo_labels == true_labels).mean()
```

이 수치가 Phase 4의 핵심 기여다.
보통 self-training 논문은 "pseudo-label이 맞는지" 검증할 수 없지만,
우리는 Philharmonia 답을 알고 있어 "모델이 얼마나 깨끗한 신호를 생성하는지" 직접 측정 가능하다.

### 매 Iteration 기록 항목

| 항목 | 의미 |
|---|---|
| 채택률 (accepted / Pool A 전체) | self-training 흡수율 |
| 클래스별 채택 수 | 다수 클래스 쏠림 발생 여부 |
| **Pseudo-label 정확도 (클래스별)** | 이 실험의 핵심 — Phase 1에서 측정 불가했던 것 |
| Pool B Macro F1 | 단계 4-3 기준선 대비 향상폭 |

---

## Pool B 재평가 (단계 4-5)

Self-training으로 보강된 MERT head를 Pool B에 재적용.

| 시점 | Pool B Macro F1 |
|---|---|
| Zero-shot (단계 4-3) | cross-domain 기준선 |
| Self-training 후 (단계 4-5) | 향상폭 |

"진짜 독립 도메인 데이터로 self-training을 했더니 cross-domain F1이 X → Y로 개선됐다"가 논문 결과.

---

## Phase 3 연결 — Spectral Flux 교차 도메인 일관성

Phase 3에서 Spectral Flux가 OrchideaSOL 기준 기법 분류의 결정적 특징임을 확인했다.
Pool B 평가 시 SVM v1을 함께 쓴다면 SHAP을 Pool B 샘플에도 적용해 다음을 확인할 수 있다:

- OrchideaSOL에서 중요했던 Spectral Flux가 Philharmonia 샘플에서도 같은 방향으로 작동하는가
- 도메인이 달라져도 음향학적 결정 근거가 일치한다면 Phase 3의 해석 신뢰도가 높아진다

이 분석은 선택 사항이며, Pool B 샘플에 SVM v1 + SHAP을 추가로 돌리면 된다.

---

## (선택) k-Shot 도메인 적응 곡선 (단계 4-6)

k = 0, 5, 10, 20개 Pool B 실제 라벨을 head fine-tuning에 추가하며 F1 변화 측정.

- k=0: pure zero-shot (단계 4-3 결과와 동일)
- k 증가: "새 도메인 적응 비용" 정량화
- "새 도메인에 적응하려면 라벨 몇 개 필요한가"라는 독립 질문

**주의**: k-shot fine-tuning에 Pool B를 쓰면 Pool B가 더 이상 순수 holdout이 아니다.
k-shot에는 Pool B의 일부(k개)만 사용하고, 평가는 나머지로만 한다 — 또는 별도 split을 만든다.

---

## 결과 저장 위치

```
results/phase4/
  report.md
  poolB_zeroshot_confusion.png       ← 단계 4-3 (MERT, SVM, CNN 각각)
  pseudolabel_accuracy_by_class.png  ← 단계 4-4
  poolB_selftraining_confusion.png   ← 단계 4-5
  kshot_curve.png                    ← 단계 4-6 (선택)
data/
  philharmonia_pool_a.json
  philharmonia_pool_b.json
  philharmonia_mert_embeddings.npy
  philharmonia_mert_labels.npy
  philharmonia_mert_meta.csv
```

---

## 주요 위험 요소

| 위험 | 대응 |
|---|---|
| Philharmonia MP3 MERT 임베딩 추출 시간 | 1회 추출 후 캐싱. Pool A+B 전체 한 번에 처리 |
| sul_ponticello Pool B 샘플 수 부족 (최대 13개) | per-class 결과 해석 시 주의 문구 추가 |
| MP3/WAV 코덱 차이로 특징값 분포 이동 | librosa `sr=16000` 통일, 단계 4-3에서 per-class F1 분포 확인 |
| Pseudo-label이 ordinario에 쏠림 | 클래스별 채택 상한 (ordinario 채택 수 ≤ 다른 클래스 채택 합) |
| Pool B leakage | Pool B는 단계 4-3, 4-5 평가에만 사용. k-shot(4-6) 시 별도 처리 |
| Self-training 후 F1 하락 | pseudo-label 정확도 기록으로 원인 추적. τ 높여서 재실행 |
