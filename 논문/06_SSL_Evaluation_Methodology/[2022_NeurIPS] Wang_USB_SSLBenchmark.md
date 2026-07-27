[2022_NeurIPS] USB: A Unified Semi-supervised Learning Benchmark for Classification

저자: Yidong Wang, Hao Chen, Yue Fan, Wang Sun, Ran Tao, Wenxin Hou, Renjie Wang, Linyi Yang, Zhi Zhou, Lan-Zhe Guo, Heli Qi, Zhen Wu, Yu-Feng Li, Satoshi Nakamura, Wei Ye, Marios Savvides, Bhiksha Raj, Takahiro Shinozaki, Bernt Schiele, Jindong Wang, Xing Xie, Yue Zhang
출판: NeurIPS 2022 (Datasets and Benchmarks Track) / arXiv:2208.07204
링크/DOI: https://arxiv.org/abs/2208.07204
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
기존 SSL 벤치마크가 CV 태스크에만 편중되고 학습 비용이 과도하다는 문제를 해결하기 위해, CV/NLP/오디오 15개 태스크에 걸쳐 14개 SSL 알고리즘을 통합 평가하는 USB 벤치마크를 제안한다.

2. 핵심 내용 & 주요 기여
- 기존 TorchSSL은 5개 CV 태스크에 335 GPU days; USB는 15개 태스크에 단 39 GPU days.
- pre-trained ViT(CV), BERT(NLP), Wav2Vec 2.0/HuBERT(Audio)를 backbone으로 활용해 학습 비용 대폭 절감.
- CV 5개, NLP 5개, 오디오 5개(GTZAN, UrbanSound8k, ESC-50, FSDnoisy18k, Keyword Spotting) 태스크 포함.
- 14개 SSL 알고리즘(FixMatch, FlexMatch, FreeMatch, SimMatch 등) 구현 및 오픈소스 공개.
- 핵심 발견: (1) 다양한 도메인이 SSL 평가의 포괄성을 높임; (2) pre-training이 SSL 성능과 일반화에 효과적; (3) 레이블 데이터가 적을수록 unlabeled data가 항상 도움이 되는 것은 아님.

3. 한계 / 비판점
- 오디오 태스크가 음악보다는 범용 sound event/speech 위주 (GTZAN 제외하면 음악 특화 태스크 없음).
- ViT/BERT 같은 대형 backbone 의존; 소형 모델 환경에서의 SSL 비교는 제한.
- NLP/오디오 validation set 구성이 labeled data에서 샘플링되어 실제 SSL 환경과 다소 다름.
- 알고리즘 간 비교에 Friedman rank를 사용하지만 태스크 특성 차이로 인한 편향 가능.

4. 추가로 읽을 references
- TorchSSL (Zhang et al., 2021): 기존 CV 중심 SSL 벤치마크
- FixMatch (Sohn et al., 2020): pseudo-labeling + consistency regularization의 핵심 baseline
- FlexMatch (Zhang et al., 2021): curriculum pseudo labeling
- FreeMatch (Wang et al., 2023): self-adaptive thresholding


OPTIONAL

A. 방법론
- SSL 패러다임: pseudo-labeling + consistency regularization 기반
- 14개 알고리즘: Π-Model, Pseudo Labeling, Mean Teacher, VAT, MixMatch, ReMixMatch, UDA, FixMatch, Dash, CoMatch, CRMatch, FlexMatch, AdaMatch, SimMatch
- 평가: Friedman rank 기반 알고리즘 순위; 레이블 수 설정별 2~3 seed 평균

B. 데이터셋
- CV: CIFAR-100, STL-10, EuroSAT, TissueMNIST, Semi-Aves
- NLP: IMDB, AG News, Amazon Review, Yahoo! Answer, Yelp Review
- Audio: GTZAN(음악 장르), UrbanSound8k, ESC-50, FSDnoisy18k, Keyword Spotting

C. 결과
- SimMatch, FreeMatch가 전반적으로 상위권
- CV 태스크와 달리 NLP/오디오에서는 알고리즘 간 성능 격차 작음
- pre-training 사용 시 레이블 효율성 크게 향상

D. 내 프로젝트(string-playing-technique-classification)와의 비교
유사점: 레이블 데이터가 적은 상황에서 SSL 알고리즘으로 성능 향상을 노리는 구조; 오디오 분류 태스크 포함.
차이점: 현악기 기법 분류는 USB의 일반 sound event 분류보다 세밀한 음색 차이를 다루며, 클래스 불균형이 더 심각할 수 있음. GTZAN(10 class 음악 장르)은 현악기 기법(수십 class)보다 훨씬 단순.
적용 인사이트: USB의 오픈소스 코드베이스를 현악기 기법 데이터에 직접 적용 가능. pseudo-labeling + mixup 조합, self-adaptive thresholding(FreeMatch) 전략이 소규모 레이블 환경에서 유용할 것.
