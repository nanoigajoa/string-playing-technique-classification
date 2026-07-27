# Phase 2 — Audio-Language Model 환각(Hallucination) 평가

## 실제 결과 (2026-06-30 완료)

> 설계 문서는 아래. 실행 결과 요약은 여기.

- 모델: Qwen2-Audio-7B-Instruct (float16, device_map=auto, MPS)
- 평가셋: 125 샘플(25/class), Y/N + open-ended = 250 쿼리

| 질문 유형 | 정확도 |
|---|---|
| Yes/No | **50.4%** (랜덤 수준) |
| Open-ended | **20.0%** (5-class 랜덤 수준) |

- pizzicato_secco만 open 100% 정확 — 나머지 4클래스 모두 0%
- ordinario Y/N miss 84% — 가장 큰 환각
- Phase 1/2 교집합 없음: MERT 최악(sul_p/col_l) ≠ ALM 최악(ordinario/tremolo)
- 결과 파일: `results/phase2/` (report.md, misclassification_heatmap.png)
- 스크립트: `src/phase2_alm_eval.py`

---

## 목적

대형 Audio-Language Model(ALM)이 화성·구조·장르처럼 학습 데이터가 풍부한 영역은 잘 처리하지만,
보잉 주법처럼 전문가 라벨링이 필요한 미세 영역에서는 체계적인 한계를 보인다는 것을 실증한다.

이 Phase의 핵심은 모델 사양이 아니라 **"한계를 드러내는 평가 설계"** 자체다.
Phase 1에서 self-training이 어려워했던 클래스와 ALM이 틀리는 클래스가 겹치는지가
가장 중요한 관찰이다.

---

## 왜 Qwen2-Audio-7B-Instruct인가

원래 계획의 nvidia/audio-flamingo-3-hf는 사용 불가:
- 8B 파라미터, BF16, 공식 테스트 하드웨어 A100 80GB
- CUDA 전용 — MPS 지원 없음, NVIDIA Ampere/Hopper 전용 연산 사용

**Qwen2-Audio-7B-Instruct 선택 근거:**
- 8B 파라미터, BF16 → 가중치 약 16GB, 24GB M5에서 float16 추론 가능
- Apache 2.0 라이선스 — 학술/포트폴리오 목적 제한 없음
- `device_map="auto"` → MPS 백엔드로 자동 라우팅
- Audio-Text-to-Text 태스크 — 오디오 입력 + 텍스트 질문 → 텍스트 응답 구조
- 한국어/영어 이중 언어 지원 (질문은 영어로 작성 권장, 평가 일관성을 위해)

### 대안 모델 (Qwen2-Audio가 로드 불가일 경우)

| 대안 | 크기 | 특징 |
|---|---|---|
| Qwen2.5-Omni-7B | 7B | 더 최신, 오디오+텍스트 지원 |
| SALMONN-7B | 7B | 음악 이해 특화 |

---

## 모델 로드 전략 (단계 2-1)

### 메모리 계획

```
BF16 로드: 8B × 2 bytes = ~16GB 가중치
+ KV cache, activations ≈ 3~5GB
총 ≈ 19~21GB (24GB 한계에서 여유 약 3~5GB)
```

### 로드 순서

1. **1차 시도**: `torch_dtype=torch.float16, device_map="auto"` — MPS 라우팅 시도
2. **2차 시도 (OOM)**: `optimum-quanto` 라이브러리로 int8 양자화
   - bitsandbytes는 CUDA 전용으로 MPS 불가 — 절대 사용하지 말 것
   - `from optimum.quanto import quantize, qint8` 사용
3. **3차 시도 (여전히 불가)**: CPU 추론 — 속도 매우 느리지만 125개 샘플은 허용 범위

### 30초 청크 처리

OrchideaSOL 샘플은 대부분 3~8초로 짧다.
Qwen2-Audio의 최대 입력 길이는 30초 기준으로 설계되어 있으므로
단일 샘플을 그대로 입력하면 된다. 10분 오디오를 청크로 나누는 처리는 이 프로젝트에서 불필요.

---

## 평가 세트 설계 (단계 2-2)

### 설계 원칙

- OrchideaSOL 정답 라벨을 활용해 **코드로 자동 생성** — 외부 API, GPT 호출 불필요
- 클래스별 균형 샘플링: 각 기법 25개, 총 125개
- 각 샘플에 질문 2종(yes/no, open-ended) 모두 적용 → 총 250개 질의

### 질문 템플릿

#### yes/no 형식

```
"This is a recording of a {instrument} playing a single note.
 Does this audio contain {technique} bowing technique?
 Answer with only 'yes' or 'no'."
```

`{technique}`에 해당 기법의 정답을 직접 물어보는 경우와,
**다른 기법을 물어보는 오답 유도** 경우를 구분해서 설계한다:

| 질문 유형 | 정답 라벨 | 질문 속 기법 | 기대 정답 |
|---|---|---|---|
| True positive 유도 | pizzicato_secco | pizzicato | yes |
| False positive 유도 | ordinario | spiccato | no |

두 유형을 섞어야 "모델이 항상 yes라고 대답하는지"를 잡아낼 수 있다.

#### open-ended 형식

```
"This is a recording of a string instrument playing a single note.
 Describe the bowing technique used in this audio.
 Be specific about whether you hear arco, pizzicato, tremolo, sul ponticello,
 or col legno battuto."
```

클래스 이름을 모두 나열한 closed-set 버전과,
클래스 이름을 주지 않는 open-set 버전 두 가지를 실험한다.

### 악기 표기 방식

OrchideaSOL 파일명에서 악기 코드를 추출해 자연어로 변환:
- Vn → violin
- Va → viola
- Vc → violoncello
- Cb → contrabass

### 저장 형식

`data/eval_set.json`:
```json
[
  {
    "file": "Vn-piz-A4-mf-1c-N.wav",
    "true_label": "pizzicato_secco",
    "instrument": "violin",
    "question_yn": "Does this audio contain pizzicato bowing technique? ...",
    "question_yn_label": "yes",
    "question_open": "Describe the bowing technique used ...",
    "expected_keyword": "pizzicato"
  },
  ...
]
```

---

## 추론 실행 (단계 2-3)

### 배치 처리

Qwen2-Audio는 단일 오디오+텍스트 입력을 한 번에 처리한다.
M5에서 메모리 안정성을 위해 **한 번에 1개씩** (batch_size=1) 처리하고,
매 10개 처리마다 MPS 캐시를 비워주는 것이 권장된다.

### 응답 파싱

#### yes/no 응답 추출

모델이 "Yes, this audio contains..." 처럼 자유롭게 답변하므로 정규식으로 추출:

```
1순위: 응답 첫 단어가 "yes" / "no"인지 확인
2순위: 응답 전체에서 "yes" / "no" 등장 횟수로 판단
3순위: 파싱 실패 → "ambiguous"로 기록 (별도 집계)
```

파싱 실패율 자체도 지표로 기록한다. 특정 기법에서 모델이 일관되게 애매한 답변을 내놓는다면
그것도 환각의 한 형태다.

#### open-ended 응답 채점

정답 기법 키워드가 응답에 포함되는지 확인:

| 기법 | 탐지 키워드 |
|---|---|
| ordinario | arco, ordinary, normal, standard, regular |
| pizzicato_secco | pizzicato, pluck, pizz |
| tremolo | tremolo, trembling, rapid |
| sul_ponticello | ponticello, bridge, sul pont |
| col_legno_battuto | col legno, wood, stick, battuto |

키워드 없이 엉뚱한 기법을 언급하면 오답으로 처리.
복수 기법을 나열하면 "정답 포함 여부"로만 채점하고,
"정답만 언급"과 "정답 포함 + 오답 추가"를 별도 구분.

---

## 지표 정의 (단계 2-4)

### 주법별 오인식률 (Misclassification Rate)

```
오인식률(기법 A) = (기법 A 샘플에서 틀린 응답 수) / (기법 A 총 샘플 수)
```

yes/no와 open-ended 각각 계산. 히트맵으로 시각화.

### Confusion pattern

yes/no 형식에서:
- "정답을 물었을 때 no라고 한 비율" (False Negative Rate)
- "오답을 물었을 때 yes라고 한 비율" (False Positive Rate)

이 두 지표가 기법별로 다른 패턴을 보이는지 확인.

### 캘리브레이션

모델이 응답에 표현한 확신("I'm confident that...", "possibly...", "I think...")과
실제 정답 여부 간의 상관관계. 텍스트 기반 확신 표현을 3단계(high/medium/low)로 분류.

> "모델이 확신 있게 틀리는지, 아니면 불확실하게 틀리는지"가 환각 성격을 구분하는 핵심

### Phase 1과의 교집합 분석

Phase 1 완료 후 "seed 비율 전반에 걸쳐 consistently 낮은 F1 클래스"를 목록으로 뽑는다.
Phase 2에서 오인식률 상위 클래스와 비교.

```
Phase 1 실패 상위 클래스 ∩ Phase 2 오인식률 상위 클래스
    → Phase 3 케이스 스터디 대상
```

이 교집합이 크면 "주법 분류의 본질적 어려움이 모델 종류를 초월한다"는 주장이 성립한다.
교집합이 작으면 "두 모델이 서로 다른 지점에서 실패한다"는 보완적 관계의 발견이 된다 — 
이것도 유의미한 결과다.

---

## 리포트 구성

`results/phase2/report.md`:
1. 모델 로드 설정 (dtype, device, 메모리 사용량)
2. 평가 세트 통계 (클래스 분포, 질문 유형별 수)
3. 오인식률 히트맵 (기법 × 질문 유형)
4. Calibration 분석
5. Phase 1 교집합 분석 — "두 모델이 공통으로 어려워하는 클래스"
6. "ALM이 주법 미세 분류에서 보이는 실패 패턴" 요약

---

## 주요 위험 요소

| 위험 | 대응 |
|---|---|
| Qwen2-Audio MPS OOM | optimum-quanto int8 fallback, 최후에는 CPU 추론 |
| 응답 파싱 실패율 높음 | 파싱 실패 자체를 지표로 기록, 수동 검수 30개 |
| 모델이 클래스 이름 모름 | "sul ponticello"를 "sul pont"로도 탐지, 유사 표현 키워드 목록 확장 |
| 125개 샘플이 너무 적음 | 필요 시 클래스별 50개(총 250개)로 확장 — 추론 시간 확인 후 결정 |
| 모델 결과가 너무 좋은 경우 | 클래스 이름 없는 open-set 질문으로 재평가, 또는 유사 기법 혼동(tremolo vs spiccato)에 집중 |
