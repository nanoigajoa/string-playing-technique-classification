[2020_IJCNN] Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning

저자: Eric Arazo, Diego Ortego, Paul Albert, Noel E. O'Connor, Kevin McGuinness
출판: IJCNN 2020 / arXiv:1908.02983 (cs.CV)
링크/DOI: https://arxiv.org/abs/1908.02983
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
단순한 pseudo-labeling이 confirmation bias(잘못된 pseudo-label을 점점 확신하는 문제)로 인해 실패함을 보이고, mixup augmentation과 labeled sample mini-batch 오버샘플링이 이를 효과적으로 완화해 consistency regularization 기반 SOTA와 동등한 성능을 달성함을 제안한다.

2. 핵심 내용 & 주요 기여
- 기존 pseudo-labeling의 한계: hard pseudo-label은 confidence bias를 피드백 루프로 강화 → confirmation bias.
- 해결책: (1) soft pseudo-label (softmax 출력값 그대로 사용); (2) mixup augmentation (예측 calibration, bias 완화); (3) minimum labeled samples per mini-batch k (correct supervision 상시 유지).
- 추가 네트워크, 그래프 구조, 복수 forward pass 없이 단순한 pseudo-labeling만으로 SOTA 달성.
- CIFAR-10/100, SVHN, Mini-ImageNet에서 Mean Teacher, UDA 등 consistency regularization 방법과 동등.
- 손실: L = L_s + L_u (soft CE) + λ_A·R_A (all-class regularization) + λ_H·R_H (entropy regularization).

3. 한계 / 비판점
- 클래스 균형(balanced) 가정; 불균형 데이터에서는 R_A(uniform prior) 가정이 적절하지 않을 수 있음.
- 오디오/음악 도메인 실험 없음.
- mixup의 alpha 파라미터 민감도 별도 분석 필요.
- FixMatch, FlexMatch 등 이후 방법보다 성능 낮음 (ICLR 2020 이전 논문).

4. 추가로 읽을 references
- Mean Teacher (Tarvainen & Valpola, 2017): consistency regularization 주요 기준선
- MixUp (Zhang et al., 2018): mixup augmentation
- FixMatch (Sohn et al., 2020): pseudo-labeling + consistency regularization 통합의 후속


OPTIONAL

A. 방법론
- 입력: labeled + unlabeled 이미지
- Soft pseudo-label: ỹ = h_θ(x_i) (softmax 출력을 unlabeled target으로 직접 사용)
- Mixup: x = δx_p + (1−δ)x_q, y = δy_p + (1−δ)y_q (δ ~ Beta(α,α))
- Minimum k labeled samples per mini-batch: supervision 강도 유지
- 정규화: R_A(all-class loss) + R_H(entropy loss)
- 모델: WRN-28-2, PR-18 등

B. 데이터셋
- CIFAR-10/100 (250~4000 labels), SVHN, Mini-ImageNet

C. 결과
- CIFAR-10 (4000 labels): 6.28% error (Mean Teacher 6.28% 동등)
- Naive pseudo-labeling → mixup 적용 시 error 대폭 감소 (CIFAR-10: 11.40% → 7.16%)
- Consistency regularization 없이도 SOTA 수준 달성

D. 내 프로젝트(string-playing-technique-classification)와의 비교
유사점: 레이블 데이터가 적은 상황에서 pseudo-labeling 활용 시 confirmation bias 문제가 현악기 기법 분류에서도 발생할 수 있음 (특히 시각적으로 구분 어려운 기법들이 초기에 오분류될 경우).
차이점: 오디오 입력에서는 mixup이 시간/주파수 영역에서의 혼합을 의미하므로 음악 신호에 물리적으로 의미 있는 방식으로 적용 필요.
적용 인사이트: audio mixup(SpecMix, waveform mixup)을 soft pseudo-labeling과 결합하면 현악기 기법 semi-supervised 학습에서 confirmation bias 완화 가능. minimum labeled samples per batch 전략은 구현이 단순해 즉시 적용 가능.
