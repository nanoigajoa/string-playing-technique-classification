[2018_NeurIPS] Realistic Evaluation of Deep Semi-Supervised Learning Algorithms

저자: Avital Oliver, Augustus Odena, Colin Raffel, Ekin D. Cubuk, Ian J. Goodfellow
출판: NeurIPS 2018 / arXiv:1804.09170 (cs.LG)
링크/DOI: https://arxiv.org/abs/1804.09170
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
기존 SSL 벤치마크가 현실적인 조건(고품질 supervised baseline, class distribution mismatch, 소규모 validation set 등)을 무시해 SSL의 실제 효용을 과장한다고 주장하며, 통합 reimplementation과 현실적 평가 조건 하에서의 재실험을 통해 SSL의 실제 한계를 밝힌다.

2. 핵심 내용 & 주요 기여
- 기존 평가 문제 6가지 제시: (P1) 공유 구현 없음, (P2) 저품질 supervised baseline, (P3) transfer learning 비교 없음, (P4) class distribution mismatch 미고려, (P5) labeled/unlabeled 데이터 양 변화 미고려, (P6) 너무 큰 validation set.
- 동일 구현 하에서 고품질 supervised baseline이 SSL 방법 대부분을 이기거나 동등함.
- Unlabeled data가 labeled data와 다른 클래스 분포(OOD)를 가질 때 SSL 성능이 크게 저하됨.
- 각 SSL 방법은 labeled/unlabeled 데이터 양에 대한 민감도가 크게 다름.
- Google Brain에서 모든 방법을 단일 코드베이스로 reimplementation하여 공개.

3. 한계 / 비판점
- 2018년 기준 방법들(Π-Model, Mean Teacher, VAT, EntMin, Pseudo-Label)만 평가; FixMatch 이후 SOTA는 미포함.
- 이미지 분류(CIFAR-10, SVHN)에 한정.
- "realistic"의 정의가 주관적이며, 실제 산업 현장의 다양한 조건을 모두 포괄하지는 못함.

4. 추가로 읽을 references
- Π-Model (Laine & Aila, 2017): consistency regularization 기초
- Mean Teacher (Tarvainen & Valpola, 2017): 안정적 consistency regularization
- VAT (Miyato et al., 2019): virtual adversarial training
- FixMatch (Sohn et al., 2020): 본 논문 이후 등장한 SOTA pseudo-labeling 방법
- USB (Wang et al., 2022): 이 논문의 정신을 계승한 통합 벤치마크


OPTIONAL

A. 방법론 (실험/방법론 논문)
- 입력: 이미지 (CIFAR-10, SVHN)
- 5개 SSL 방법 통합 reimplementation: Π-Model, Mean Teacher, VAT, EntMin, Pseudo-Labeling
- 동일 13-layer CNN backbone, 동일 하이퍼파라미터 탐색 예산 (1000 trials) 사용
- 평가 조건: labeled 수 변화, validation set 크기 변화, class distribution mismatch (labeled vs. unlabeled OOD) 포함

C. 결과
- 고품질 supervised baseline이 많은 SSL 방법을 능가하거나 동등
- Class distribution mismatch(unlabeled에 OOD 클래스 포함) 시 모든 SSL 방법 성능 크게 저하
- SSL 방법들은 labeled 데이터 양에 민감도가 상이

D. 내 프로젝트(string-playing-technique-classification)와의 비교
유사점: 현악기 기법 데이터는 레이블이 적고, unlabeled 오디오가 다른 악기/연주자에서 올 수 있어 class distribution mismatch 문제 직접적으로 해당.
차이점: 이미지 분류 한정 실험; 오디오 domain에서의 검증 필요.
적용 인사이트: SSL 실험 설계 시 (1) 고품질 supervised baseline을 먼저 구축하고 비교, (2) unlabeled 데이터의 클래스 분포를 확인/통제, (3) validation set 크기가 신뢰할 만한 크기인지 확인. 이 3가지는 현악기 기법 SSL 실험 설계의 필수 체크리스트.
