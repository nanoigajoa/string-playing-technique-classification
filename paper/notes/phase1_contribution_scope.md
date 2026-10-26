# 연구 기여와 주장 범위 — 2026-10-01

## 연구 질문

OrchideaSOL의 네 현악기 종류에서 한 종류 전체를 보류했을 때 제한 라벨
MERT 헤드와 245차원 음향 특징 SVM의 비교가 어떻게 달라지는가? 의사 라벨의
target 전달과 검증·선택·저장 경로를 추적한 자기학습은 어떤 결과를 보이는가?

## 현재 자료로 뒷받침하는 기여

| 기여 | 확인한 증거 | 주장 범위 |
|---|---|---|
| 의사 라벨 target 소유권을 명시한 감사 사례 | 관측/채택 표본과 target 쌍, 반복별 풀 해시, 감독·의사 라벨·oracle 진단의 48개 대응 분할 | 저장 기록의 target 경로를 검증한 구현 사례. 의사 라벨이 항상 개선한다고 주장하지 않음 |
| 모델 선택에서 보고 수치까지 연결한 평가 자료 | MERT/SVM 48쌍·CNN 4개·자기학습 144개의 checkpoint 재실행, 저장 예측·지표·해시 대조 | 해당 로컬 실행의 결과 출처. 공개 묶음은 저장 예측의 통계를 재계산하며 checkpoint를 재실행하지 않음 |
| 제한 라벨 악기종 보류 비교의 조건부 결과 | 각 예산의 세 trial 평균을 먼저 구한 네 악기종 평균 | MERT/SVM 우열이 예산에 따라 바뀌고, 자기학습 5% 평균은 감독학습보다 낮음 |
| 같은 예측에서 재표집 단위를 바꾼 민감도 | 대응 clip·파일명 proxy·악기종 재표집 각각 1,000회 | 측정 단위에 따른 구간 변화. 새 연주자 모집단의 신뢰구간이나 모델 동등성은 아님 |
| 모델 파일 없이 재계산할 수 있는 결과 자료 | 예측 100파일·244개 모델 예측 벡터에서 표 10개를 다시 계산하고 대조하는 코드 | 저장 예측의 통계적 재현. 원자료에서의 전체 재학습 재현은 별도 작업 |

## 선행 연구와의 관계

특징/SVM 기반 현악 주법 분류는 Kruger and Jacobs (2020), 음악 사전학습
표현은 Li et al. (2024)의 MERT와 연결된다. 이번 연구는 두 방식의 자체
부분집합·자체 분할 비교를 제공한다. 선행 논문의 과제와 프로토콜이 달라
발표 성능과 직접 비교하거나 새로운 분류 방법의 우수성을 주장하지 않는다.

Oliver et al. (2018)의 감독학습 대조군 논의, Kapoor and Narayanan (2023)의
누수 검토, Rodríguez-Algarra et al. (2019)의 음악 분류 교란 분석을 배경으로
평가 절차를 구체적인 주법 분류 실행에서 추적한다. 이번 구현 수정 자체를
새로운 일반 평가 이론으로 제시하지 않는다. 참고문헌의 전체 서지는 원고에 있다.

## 원고에 사용할 기여 문장

**국문:** 본 연구는 다섯 현악 주법의 악기종 보류 평가에서 관측 라벨과
의사 target의 전달 경로, 학습 분할에서 적합한 통계량, 검증으로 선택한
checkpoint와 저장 예측을 연결한 감사 사례를 제시한다. 제한 라벨 조건의
MERT/SVM 비교와 설정을 맞춘 자기학습 진단을 보고하고, 고정 예측에 대한
재표집 단위를 바꿔 추론 범위의 민감도를 확인한다. 결과 자료는 모델 파일 없이
재계산할 수 있으며, 결론은 관찰된 네 악기종에 조건부이다.

**English:** We present an auditable case study of five string playing techniques
under instrument-type holdout. The evaluation traces observed labels and
pseudo-target ownership, train-local fitted quantities, validation-selected
checkpoints and persisted predictions. It reports limited-label MERT/SVM
comparisons and matched self-training diagnostics, with resampling-unit
sensitivity on fixed predictions. The supplied statistics can be recomputed
without model artifacts; conclusions remain conditional on four observed types.

## 철회하거나 보류할 주장

- 과거 실행 간 차이를 특정 결함 하나의 인과 효과로 해석하는 주장: 동시에 바뀐
  구현·설정·표본·선택 경로를 분리할 통제 실험이 없다.
- 새 연주자·물리적 악기·녹음 세션 일반화: 검증된 식별자가 없다.
- MERT/SVM 동등성: 사전 허용 격차와 충분한 독립 그룹이 없다.
- 독립 사전등록 확인 실험: 자료를 이미 탐색한 뒤의 canonical 재실행이다.
- Layer-weighted probe·LoRA 순위, SHAP 기전, 외부 전이: 같은 검증 범위를
  완료하지 않아 현재 기여에서 제외한다.

## 학술지 연구 논문으로 강화할 수 있는 증거

현재 자료는 감사 사례의 기술적 추적성과 조건부 비교를 제공한다. 연구 논문에
충분한 새로운 기여가 있는지는 별도로 검토해야 한다. 강화 작업을 시작한다면
다음 중 연구 질문을 정하고, 테스트 자료를 보기 전에 분석 계획을 고정한다.

1. 같은 분할·시드·예산에서 결함 하나만 바꾼 통제 실험으로 원인별 영향을 분리한다.
2. 실제 연주자·세션 식별자를 갖춘 독립 자료로 같은 주법 분류 질문을 평가한다.
3. 다른 데이터/모델에서도 target 경계 감사가 반복되는지 적용 범위를 검토한다.

이 문서는 추가 실험을 완료했다고 주장하거나 새 모델 결과를 포함하지 않는다.
