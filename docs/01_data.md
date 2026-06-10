# 데이터 확보 · 클래스 설계 · 분할

## Phase 1 — 데이터 확보 및 클래스 설계

### 작업 순서
1. Ircam Forum 가입 → OrchideaSOL 다운로드 (arXiv 2007.00763)
2. 현악 기법별 샘플 수 카운트
3. 4~5개 클래스 확정 — 쉬운 기법과 어려운 기법을 혼합

### 클래스 설계 원칙

- **쉬운 축:** Arco vs Pizzicato (어택 엔벨로프만으로 분리됨)
- **어려운 축:** Spiccato / Détaché / Martelé / Staccato, Tremolo vs Arco
- 샘플 수 200개 미만 클래스는 통합 또는 제외

> 쉬운 기법만 쓰면 99% 정확도가 나오고 배울 게 없다. 어려운 보잉 아티큘레이션 클래스를 반드시 포함할 것.

### 데이터셋 주의사항

| 데이터셋 | 상태 | 비고 |
|---|---|---|
| **OrchideaSOL** | 사용 | 52종 연주기법 라벨, Ircam Forum 가입 필요 |
| **TinySOL** | 사용 불가 | ordinario(평주법)만 포함 — 기법 분류에 무용지물 |

---

## Phase 3 — 데이터 분할 및 누수 방지

### 원칙

같은 녹음/같은 악기의 다른 음이 train과 test에 동시에 들어가면 정확도가 과대평가된다.

### 방법

- **OrchideaSOL 공식 5-fold 분할** 메타데이터를 그대로 사용 (악기·기법·음역·강약 균형 보장)
- 직접 분할이 필요한 경우:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=recording_id):
    ...
```

> `groups`는 악기/녹음 단위로 지정. `random_state`만 맞춰도 소용없다 — 반드시 그룹 단위 분할.
