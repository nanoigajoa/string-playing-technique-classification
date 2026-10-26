# v0 원고 주장 대조 (2026-09-28)

대상: `연구과정/주법분류_논문초안_국문_v0.md`, `연구과정/주법분류_논문초안_영문_v0.md`. 두 문서는 역사적 초안으로 표시했다. README는 canonical 결과 중심으로 개편하고, 국·영문 v1 초안을 새로 작성했다. 아래 항목은 v0에 남은 주장으로, v1에서는 교체하거나 제외했다.

| 위치/주장 | 판정 | 필요한 수정 |
|---|---|---|
| 초록·README: “결론이 두 번 뒤집힘”, 자기학습에서 누수 제거로 5% `0.745 → 0.638` | **인과 효과 미입증** | 기존 경로에는 test 선택과 숨은 정답 학습이 함께 있었다. 수정된 동일 분할·시드·고정 τ=0.8의 5% 결과는 supervised 0.6686, pseudo 0.6364, 명시적 oracle 0.6489이다. 과거 수치와의 차이를 test 선택 효과로 해석하지 않는다. |
| §4.2 튜닝 MERT 대 SVM 표 | **수치 불일치** | 새 MERT/SVM fold 평균은 5% `0.6655/0.6613`, 10% `0.7634/0.7542`, 20% `0.8242/0.8316`, 50% `0.8599/0.8681`이다. 같은 저장 예측에서 재생성된 `../mert_svm/canonical_summary.csv`를 사용한다. |
| §4.3 기존 BCa 구간 및 “세 번째 반전” | **대상 checkpoint 불일치** | 과거 bootstrap은 최종 튜닝 표의 저장 checkpoint와 동일하지 않았다. 새 `bootstrap_sensitivity.csv`는 canonical 예측과 대응된다. clip·파일명 기반 recording proxy·악기종 단위 결과를 각각 표시하고, 네 악기종의 재표집을 새 연주자 모집단의 구간으로 부르지 않는다. |
| 초록·§4.2·§4.4: “통계적으로 구분되지 않음” 또는 “동급” | **검증 불충분** | 새 악기종 재표집의 MERT−SVM 구간은 네 예산 모두 0을 포함한다. 이는 동등성의 증거가 아니다. 사전 정의한 허용 격차와 독립된 그룹 자료가 없으므로 동등성 결론을 삭제한다. |
| §4.4·§5: layer-weighted, LoRA, layer probe와 SHAP의 인과적 설명 | **역사적 탐색 결과** | 해당 학습 경로의 train-only 통계·checkpoint provenance가 canonical 경로와 연결되지 않았다. 별도 수정·재실행·예측 일치 검증 전에는 최종 비교 또는 기전의 확정 증거로 쓰지 않는다. |
| 전체라벨 CNN `0.905`와 기타 v1 수치를 동일 비교표에 배치 | **평가 경로 혼합** | canonical CNN은 네 악기종 fold 평균 `0.8729`이다. 원래 v1 수치와 새 canonical 5~50% 수치는 다른 학습 예산·선택 절차이므로 분리해서 보고한다. |

수치 원본은 `report.md`, `bootstrap_sensitivity.csv`, `per_class_f1.csv`, `confusion_counts.csv`, `../self_training/canonical_summary.csv`이다. `src/phase1_manuscript_consistency.py`가 새 국·영문 v1의 핵심 표를 검증한다. v1 DOCX는 해당 Markdown에서 생성했다. v0의 그림·DOCX는 수정하지 않았으며 제출본이 아니다.
