[2023_ICLR] FreeMatch: Self-Adaptive Thresholding for Semi-Supervised Learning

저자: Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernt Schiele, Xing Xie
출판: ICLR 2023 / arXiv:2205.07246 (cs.LG)
링크/DOI: https://arxiv.org/abs/2205.07246
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
기존 SSL의 고정 임계값(threshold) 또는 ad-hoc 조정 방식의 한계를 극복하기 위해, 모델의 학습 상태에 따라 global(dataset-level)과 local(class-level) 임계값을 자동으로 조정하는 Self-Adaptive Thresholding(SAT)과 클래스 공정성 정규화(SAF)를 제안한다.

2. 핵심 내용 & 주요 기여
- FixMatch(고정 threshold 0.95)와 FlexMatch(pre-defined global threshold 기반 class-specific 조정)의 한계: 레이블 극소 상황에서 threshold가 너무 높으면 초기 학습이 느리고 unlabeled data 활용 미흡.
- SAT(Self-Adaptive Thresholding): exponential moving average(EMA)로 전체 unlabeled 데이터의 평균 confidence를 추정 → global threshold로 사용. 각 class별 EMA probability로 local threshold 조정.
- SAF(Self-Adaptive class Fairness): 균등 prior로 예측 분포를 정규화해 초기 학습 시 특정 클래스 편향 방지.
- CIFAR-10(1 label/class): FixMatch 대비 5.78% error rate 감소; STL-10(4 labels/class): 13.59% 감소; ImageNet(100 labels): 1.28% 감소.
- 클래스 불균형 SSL에서도 성능 향상.

3. 한계 / 비판점
- EMA 기반 threshold 추정은 초기 학습에서 신뢰도 낮을 수 있음 (warm-up 필요).
- 하이퍼파라미터(EMA decay, SAF weight) 설정에 민감할 수 있으나 논문 내 sensitivity 분석 제한적.
- 오디오/음악 도메인 실험 미포함; CV 중심 벤치마크(CIFAR, STL-10, ImageNet)만 검증.
- FlexMatch와 성능 차이가 일부 setting에서 작음.

4. 추가로 읽을 references
- FixMatch (Sohn et al., 2020): 고정 threshold pseudo-labeling의 핵심 기준선
- FlexMatch (Zhang et al., 2021): class-specific curriculum threshold
- USB (Wang et al., 2022): FreeMatch가 포함된 통합 벤치마크
- AdaMatch (Berthelot et al., 2022): 또 다른 adaptive threshold 방법


OPTIONAL

A. 방법론
- 입력: labeled + unlabeled 이미지 (weak/strong augmentation)
- SAT: global_t = EMA(max_confidence over all unlabeled); local_t(c) = global_t × EMA(p(c))
- SAF: fairness loss L_f = U log E[q_b], uniform prior U로 예측 분포 정규화
- 최종 손실: L = L_s (supervised CE) + L_u (unlabeled pseudo-label CE, threshold 조건부) + L_f
- 평가 지표: error rate (%)

B. 데이터셋
- CIFAR-10/100 (1~4 labels/class), STL-10 (4~40 labels/class), ImageNet (100 labels)

C. 결과
- CIFAR-10 (1 label/class): error 5.78% (FixMatch: 11.56%)
- STL-10 (4 labels/class): error 13.59% (FixMatch: 27.55%)
- 클래스 불균형 설정에서도 안정적 성능 향상

D. 내 프로젝트(string-playing-technique-classification)와의 비교
유사점: 레이블 데이터가 적고 클래스 불균형이 있는 현악기 기법 분류 문제에 SAT+SAF 전략이 적합.
차이점: FreeMatch는 이미지 분류 검증; 오디오 특징과 함께 동작하는지 추가 검증 필요.
적용 인사이트: 현악기 기법 데이터에서 일부 기법(ex. 특수 주법)의 레이블이 극히 적을 때, 고정 threshold FixMatch보다 FreeMatch/self-adaptive 방식이 유리할 가능성. USB 코드베이스로 쉽게 적용 가능.
