# v1 canonical 원고에서 제외한 모델의 코드 감사

기준일: 2026-09-28. 아래는 재실험으로 검증한 결과가 아니라 기존 실행 경로의 정적 감사다. 이전 출력과 코드는 역사적 자료로 보존한다.

| 경로 | 확인된 경계 문제 | canonical 복귀 조건 |
|---|---|---|
| `src/phase1_layer_probe.py` | 모듈 import 시 전체 `labels`의 빈도로 `class_weights`를 계산한다. `search()`는 검증 F1이 개선될 때마다 테스트셋을 평가한다. 설정 선택 조건문 자체는 검증 F1을 비교하지만, 테스트 조회가 탐색 과정에 섞여 있다. 선택 checkpoint·테스트 예측·해시를 연결하는 manifest가 없다. | 각 분할의 `tr_idx`에서 class weight를 계산하고, 모든 설정·레이어 선택 후 저장 checkpoint를 다시 불러 테스트를 한 번 평가한다. 이전 결과를 덮지 않고 예측·per-class·confusion·해시를 저장한다. |
| `src/phase1_lora.py` | `counts`가 전체 `labels`에서 계산되고 각 분할의 손실 가중치로 쓰인다. 에폭마다 조기종료 checkpoint 상태를 메모리에 보관하지만 선택 상태를 영구 저장하지 않으며 테스트 예측 파일도 남기지 않는다. 기존 CSV만으로 예측과 결과를 재검증할 수 없다. | class weight를 `tr_idx`에서 계산하고 선택된 어댑터·헤드를 저장한다. 저장 상태를 재로딩해 테스트를 한 번 평가하고 예측·해시·입력 버전을 manifest에 기록한다. |

기존 LoRA 기록의 분할별 실행 시간은 약 31–65분이다(`results/phase1/lora_results.csv`). 따라서 이 결과를 재검증 없이 v1에 포함하지 않았다. Layer-weighted와 LoRA 우열, 깊이가 도움이 되지 않는다는 기전, LP-FT 원인 설명은 새 국·영문 canonical v1에서 제외했다. 기존 표의 숫자가 참인지 거짓인지 이 정적 감사만으로는 결론낼 수 없다.
