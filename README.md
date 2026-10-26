# 현악 주법 분류 평가 감사

OrchideaSOL 현악 부분집합의 다섯 주법을 분류하면서 라벨 접근, 학습 데이터 경계, 모델 선택, 결과의 추론 범위를 검증한 연구 저장소입니다. 데이터는 바이올린·비올라·첼로·콘트라베이스의 **3,402개 클립**입니다. 외부 평가는 한 번에 악기 **종류** 하나를 보류하는 네 fold로 수행합니다.

현재 국·영문 원고는 [국문 canonical v1](연구과정/주법분류_논문초안_국문_v1_canonical.md)과 [영문 canonical v1](연구과정/주법분류_논문초안_영문_v1_canonical.md)입니다. 같은 내용의 [국문 DOCX](연구과정/주법분류_논문초안_국문_v1_canonical.docx)와 [영문 DOCX](연구과정/주법분류_논문초안_영문_v1_canonical.docx), [영문 검토용 PDF](results/phase1/canonical/manuscript/review_preview.pdf)를 생성했습니다. 검증된 결과에 문헌 6편과 그림 3개를 반영했습니다. 공개 재현 자료·기여 범위·최종 저자 선언 등 남은 준비는 [원고 검토 기록](results/phase1/canonical/manuscript/READINESS.md)에 있습니다. 기존 v0 원고와 `results/phase1/`의 레거시 보고서는 오류 발견 과정을 보여주는 **역사적 자료**입니다.

## 검증된 주요 결과

MERT 헤드와 245차원 특징 SVM은 같은 제한 라벨 분할에서 학습하고 별도 검증셋으로 설정을 선택했습니다. 아래 값은 세 trial을 fold 안에서 먼저 평균한 뒤 네 악기종 fold를 평균한 macro F1입니다.

| 라벨 시드 | MERT | SVM | MERT − SVM |
|---:|---:|---:|---:|
| 5% | 0.6655 | 0.6613 | +0.0042 |
| 10% | 0.7634 | 0.7542 | +0.0092 |
| 20% | 0.8242 | 0.8316 | −0.0074 |
| 50% | 0.8599 | 0.8681 | −0.0082 |

저장된 checkpoint와 예측을 재실행해 **MERT/SVM 48개 분할, CNN 4개 fold, 자기학습 144개 checkpoint**를 대조했습니다. 수정된 자기학습의 5% 조건은 supervised **0.6686**, 실제 의사 라벨 **0.6364**였습니다. 전체 라벨 CNN의 악기종 fold 평균 macro F1은 **0.8729**이며, 제한 라벨 표와는 학습 예산이 다릅니다. 세부 수치·클래스별 F1·confusion matrix·재표집 구간은 [canonical 검증 보고서](results/phase1/canonical/verification/report.md)에 있습니다.

MERT−SVM의 악기종 단위 재표집 구간은 네 예산 모두 0을 포함합니다. 이는 모델 동등성을 뜻하지 않습니다. 자료에는 독립 연주자·물리적 악기·녹음 세션 ID가 없어 그 대상에 대한 일반화도 주장하지 않습니다. 이전 원고의 자기학습 `0.745 → 0.638`을 “누수 제거의 인과 효과”로 제시한 문장, layer-weighted·LoRA를 포함한 모델 순위와 기전 설명은 [원고 주장 대조표](results/phase1/canonical/verification/manuscript_claim_audit.md)에 따라 투고용 주장으로 사용하지 않습니다.

이 데이터셋은 이전 탐색 실험에서 이미 살펴봤습니다. 이번 재실행의 학습·선택·테스트 경계는 검증했지만, 전체 연구를 독립적인 사전등록 확인 실험으로 제시하지 않습니다.

## 저장소 구성

- `src/features.py`: 245차원 음향 특징 추출.
- `src/dataset.py`, `src/model.py`: mel-spectrogram CNN 입력과 모델.
- `src/phase1_canonical_*.py`: 수정된 자기학습·MERT/SVM·CNN 실행과 저장 예측 검증.
- `results/phase1/canonical/`: checkpoint manifest, 예측, 결과표, 검증 보고서.
- `models/phase1/canonical/`: 선택된 모델 checkpoint.
- `연구과정/`: 원고와 연구 계획. v0 원고는 역사적 초안입니다.

원본 OrchideaSOL 음원과 재생성 가능한 대용량 임베딩 캐시는 Git에 넣지 않습니다.

## 실행

```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/phase1_canonical_mert_svm.py
python src/phase1_canonical_cnn.py
python src/phase1_canonical_selftraining.py
python src/phase1_canonical_verify.py --draws 1000
python src/phase1_canonical_selftraining_verify.py
python src/phase1_manuscript_consistency.py
python src/phase1_canonical_figures.py
python src/phase1_manuscript_export.py
python src/predict.py path/to/audio.wav --model cnn
```

데이터 재생성과 전체 모델 학습에는 로컬 OrchideaSOL 설치, MERT 임베딩, 충분한 실행 시간이 필요합니다. 실행 코드는 MPS를 우선 사용하며 CPU 경로도 제공합니다. `python src/phase1_canonical_selftraining.py`는 기본적으로 CPU를 쓰고 `--device mps`를 선택할 수 있습니다.

그림 생성은 저장된 canonical CSV를 읽으며 모델을 다시 학습하지 않습니다. 문서 내보내기는 로컬 `pandoc`이 필요하고 DOCX와 그림을 내장한 HTML을 생성합니다. PDF는 HTML에서 별도로 출력한 검토본입니다.

## 저장 예측으로 결과 재계산

[공개 후보 자료 묶음 ZIP](results/phase1/canonical/release/canonical-reproduction.zip)과 [SHA-256](results/phase1/canonical/release/canonical-reproduction.zip.sha256)을 준비했습니다. [별도 환경 검증 기록](results/phase1/canonical/release/VALIDATION.md)에 실제 ZIP을 풀어 실행한 결과가 있습니다. 음원·임베딩·checkpoint와 로컬 음원 절대 경로는 포함하지 않습니다. 외부 공개 주소는 아직 등록하지 않았습니다.

ZIP을 푼 `canonical-reproduction/` 폴더에서:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python -B src/phase1_release_reproduce.py
.venv/bin/python -B src/phase1_canonical_figures.py --tables-root reproduced --output-dir reproduced/figures
```

이 경로는 예측 100파일의 모델 예측 벡터 244개에서 표 10개를 재계산하고 원본과 대조하며, 관측 분할과 저장 의사 target 기록도 확인합니다. Checkpoint 추론이나 재학습은 수행하지 않습니다. 모델 파일을 재실행하는 기존 검증 코드와 이 경로의 차이를 묶음 README에 명시했습니다.

`requirements-phase1-analysis.txt`는 별도 검증 환경의 분석 패키지·전이 의존성을 고정합니다. 기존 `requirements.txt`와 현재 학습 환경의 차이는 묶음의 환경 기록에 남겼으며, 과거 학습 실행의 완전한 환경 잠금 기록을 복구했다고 주장하지 않습니다. [연구 기여와 주장 범위](docs/phase1_contribution_scope.md)는 이번 결과를 네 악기종에 조건부인 평가 감사 사례로 정리합니다.
