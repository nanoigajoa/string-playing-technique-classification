[2020_MachineLearning] A survey on semi-supervised learning

저자: Jesper E. van Engelen, Holger H. Hoos
출판: Machine Learning (2020) 109:373–440 / DOI: 10.1007/s10994-019-05855-6
링크/DOI: https://doi.org/10.1007/s10994-019-05855-6
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
반지도학습(SSL)의 이론적 기반부터 최근 딥러닝 기반 알고리즘까지 포괄하는 종합 서베이로, 새로운 분류 체계(wrapper method, unsupervised preprocessing, intrinsically SSL, transductive)를 제안하며 클러스터 가정과의 연결성을 조명한다.

2. 핵심 내용 & 주요 기여
- SSL의 핵심 가정 정리: smoothness assumption, cluster assumption, manifold assumption — 이 가정들이 성립하지 않으면 unlabeled data가 오히려 해가 됨.
- 새로운 taxonomy 제안: Inductive(wrapper method, unsupervised preprocessing, intrinsically SSL) + Transductive methods.
- Wrapper method: self-training, co-training; Unsupervised preprocessing: clustering → 레이블; Intrinsically SSL: generative model, graph-based, S3VMs.
- 최근 딥러닝 기반 SSL: consistency regularization(Π-Model, Mean Teacher), pseudo-labeling, 생성 모델(VAE/GAN) 포함.
- 클러스터 가정과 semi-supervised 클러스터링 가정의 수학적 연결 제시.

3. 한계 / 비판점
- 2019년 작성 기준 — FixMatch, MixMatch 이후 급속 발전한 최신 pseudo-labeling/thresholding 방법(FreeMatch, FlexMatch 등) 미포함.
- 오디오/음악 도메인 적용 사례가 거의 없음.
- Self-supervised learning(대조 학습 등)과의 경계가 본 서베이에서 명확하지 않음.

4. 추가로 읽을 references
- Chapelle et al. (2006): Semi-supervised learning 교재 (기초)
- Zhu & Goldberg (2009): SSL 소개 교재
- Goodfellow et al. (2017): 딥러닝 교재 (뉴럴넷 기반 SSL 배경)
- FixMatch, MixMatch: 본 서베이 이후 SOTA


OPTIONAL

A. 방법론 (서베이 논문)
- 다루는 범위: 분류(classification) 중심 SSL 방법론 전체
- 학습 방식 분류: inductive vs. transductive; wrapper vs. unsupervised preprocessing vs. intrinsic
- 핵심 기법: self-training, co-training, label propagation, S3VM, mean teacher, consistency regularization, VAE/GAN 기반 생성 모델
- 평가: 각 방법의 가정 명시 및 실험적 한계 분석
