[2022_ICLR] Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution

저자: Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang
출판: ICLR 2022 / arXiv:2202.10054 (cs.LG)
링크/DOI: https://arxiv.org/abs/2202.10054
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
사전학습 모델을 full fine-tuning하면 in-distribution(ID) 성능은 좋아지지만 out-of-distribution(OOD) 성능이 linear probing보다 나빠질 수 있으며, linear probing → fine-tuning 순차 적용(LP-FT)이 두 가지 장점을 모두 취할 수 있음을 이론과 실험으로 보인다.

2. 핵심 내용 & 주요 기여
- 10개 distribution shift 벤치마크에서: fine-tuning은 linear probing보다 평균 ID +2%p 높지만 OOD -7%p 낮음.
- 이론: 2-layer linear network에서 fine-tuning이 ID 학습 데이터에 맞추기 위해 feature extractor를 왜곡(distort)하고, OOD 일반화에 필요한 pre-trained feature를 손상시킴.
- LP-FT(linear probing then fine-tuning): linear probing으로 head를 먼저 초기화 → feature extractor가 크게 바뀌지 않아 OOD 성능 보존.
- LP-FT: fine-tuning 대비 ID +0.6%p, OOD +9.6%p (10개 벤치마크 평균).

3. 한계 / 비판점
- 이론 분석이 2-layer linear network에 한정; 실제 deep nonlinear network에서의 보장 없음.
- 분포 이동이 큰 경우(large OOD shift)에만 두드러지며, 작은 이동에서는 fine-tuning이 여전히 유리할 수 있음.
- pre-trained feature의 "품질"에 의존; 저품질 backbone(MoCo-v1)에서는 효과 감소.

4. 추가로 읽을 references
- MoCo-v2 (Chen et al., 2020): 실험에 사용된 self-supervised backbone
- CLIP (Radford et al., 2021): 실험에 사용된 또 다른 backbone
- WiSE-FT (Wortsman et al., 2022): fine-tuning과 zero-shot 가중치 혼합으로 OOD 개선하는 관련 접근


OPTIONAL

A. 방법론
- 입력: 이미지 (분류 태스크)
- 설정: pretrained backbone (MoCo-v2, CLIP) 위에 (a) fine-tuning, (b) linear probing, (c) LP-FT 비교
- 이론: 2-layer linear overparameterized regression에서 fine-tuning의 OOD error 하한 분석
- 10개 distribution shift 데이터셋: Breeds-Living17, Breeds-Entity30, DomainNet, CIFAR→STL, CIFAR10.1, FMoW, ImageNetV2, ImageNet-R, ImageNet-A, ImageNet-Sketch

C. 결과
- Fine-tuning: avg ID 85.1%, OOD 59.3%
- Linear probing: avg ID 82.9%, OOD 66.2%
- LP-FT: avg ID 85.7%, OOD 68.9% → 두 방법 모두 능가

D. 내 프로젝트(string-playing-technique-classification)와의 비교
유사점: pre-trained SSL 모델(MERT 등)을 현악기 기법 분류에 transfer learning할 때 fine-tuning vs. linear probing 선택 문제와 직결.
차이점: 이 논문은 이미지 domain; 오디오에서 distribution shift의 양상이 다를 수 있음(녹음 환경, 악기 종류 등이 shift 요인).
적용 인사이트: MERT 같은 음악 pre-trained model을 현악기 기법 데이터에 적용할 때, full fine-tuning 전에 linear probing으로 head를 먼저 초기화하는 LP-FT 전략이 OOD 악기/녹음 환경 일반화에 유리할 수 있음. 실험 설계 시 ID/OOD 분할을 명시적으로 설정해 비교 필요.
