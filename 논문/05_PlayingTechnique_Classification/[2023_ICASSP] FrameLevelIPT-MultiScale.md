[2023_ICASSP] Frame-Level Multi-Label Playing Technique Detection Using Multi-Scale Network and Self-Attention Mechanism

저자: Dichucheng Li, Mingjin Che, Wenwu Meng, Yulun Wu, Yi Yu, Fan Xia, Wei Li
출판: ICASSP 2023
링크/DOI: arXiv:2303.13272
읽은 날: 2026-06-22
상태: Done

CORE

1. 한줄 요약
4번에서 읽은 Guzheng 연구팀의 후속 연구로, 합성 데이터의 한계를 넘어 실제 99개 Guzheng 솔로 녹음을 직접 라벨링하고, 한 프레임에 여러 기법이 동시에 존재할 수 있는 multi-label 문제로 재정의한 뒤, multi-scale network와 self-attention으로 F1 86.54%를 달성해 기존 모델들(4번 논문 모델 포함)을 크게 앞섬.

2. 핵심 내용 & 주요 기여
처음으로 overlapping/mixed IPT(한 음에 여러 기법이 동시에 적용되거나, 다른 기법의 음이 겹치는 경우)를 다루는 frame-level multi-label 분류로 문제를 재정의함. 기존 연구(2,3,4번 포함)는 한 프레임에 한 기법만 있다고 가정하는 single-label 문제였음. 새 데이터셋 Guzheng_Tech99는 전문 연주자 2명이 실제로 연주한 99개 솔로 곡(다양한 장르)을 그대로 쓰고, 63352개 note 단위 onset/offset/pitch/IPT annotation을 수작업으로 만듦 — 합성 콘캐트네이션이 아니라 진짜 연속 연주라는 게 2,3,4번 논문과 결정적으로 다른 점.

모델은 multi-scale network(컴퓨터비전의 HRNet에서 가져온 구조)로, 시간축 해상도가 다른 3개의 수평 branch를 두고 반복적으로 down/upsampling+concat해서 정보를 교환함. 가장 coarse한 scale에는 self-attention block을 적용해서 tremolo처럼 긴 기법의 장거리 패턴까지 포착함. 결과는 F1 86.54%로, GZFNO(4번 논문 모델, 68.73%), EHFCN(3번 Erhu FCN 모델, 68.65%), 범용 CNN+Res(70.77%)를 모두 크게 앞섬. Ablation에서는 multi-scale 구조를 single-scale로 바꾸면 F1이 9.56%p나 급락해서, 이 구조 자체가 가장 핵심적인 기여임을 보임.

3. 한계 / 비판점
여전히 단일 악기(Guzheng)에만 적용했고, 저자도 violin/Chinese bamboo flute로 확장은 future work로 남김. 클래스 불균형이 극심함(plucks 11860개 vs tremolo 77개) — weighted BCE로 일부 보정하지만 confusion matrix를 보면 소수 클래스(특히 DP, PN)가 plucks로 흡수되는 경향이 여전히 큼. Note-level이 아니라 frame-level 평가만 보고하고, 실제 음악적으로 의미 있는 단위(note)에서의 정확도는 따로 검증되지 않음(저자도 note-level 확장을 future work로 언급). Self-attention/multi-scale 구조가 상당히 복잡한데, 어떤 부분이 길고 짧은 기법을 구체적으로 어떻게 잘 잡는지에 대한 세밀한 분석은 없고 ablation도 구조 단위로만 보고됨.

4. 추가로 읽을 references
[3][4] Wang, Lostanlen, Benetos, Chew, "Playing technique recognition by joint time-frequency scattering" (ICASSP 2020) / "Adaptive scattering transforms for playing technique recognition" (TASLP 2022) — 1번 논문(DLfM 2018) 저자인 Lostanlen이 참여한 scattering transform 기반 IPT 인식 후속 연구로, 어떤 악기를 다루는지 확인 후 우선적으로 읽어볼 가치가 있음.
[1] Yang, "Computational modelling and analysis of vibrato and portamento in expressive music performance" (PhD thesis, QMUL 2017) — violin vibrato/portamento 전문 연구, 결론에서 저자들이 violin으로의 확장 근거로 직접 인용함.

OPTIONAL

A. 방법론
입력은 CQT(88 bins, hop 512, 44.1kHz, 3초 단위로 클립). 모델은 multi-scale network로, 시간축 해상도가 다른 3개의 수평 branch(high/medium/low resolution)를 두고, residual block들을 거치며 반복적으로 max-pooling(downsample)과 transposed convolution(upsample) 후 concat해서 서로 다른 스케일의 정보를 융합함. 가장 coarse한 scale의 feature map에 self-attention block(Transformer 스타일, Q/K/V로 프레임 간 전역적 의존성 포착)을 적용해서 긴 기법의 장거리 패턴을 더 잘 잡음. 최종적으로 conv+sigmoid로 프레임마다 7개 IPT의 존재 여부를 독립적으로 출력(multi-label, threshold 0.5로 이진화)하고, weighted BCE loss로 class imbalance를 보정함.

B. 데이터셋
Guzheng_Tech99: 전문 연주자 2명이 실제로 연주한 99개 Guzheng 솔로 곡(Henan/Shandong/Chaozhou/Zhejiang/Shaanxi 등 다양한 장르), 총 9064.6초, 63352개 note 단위 annotation(Sonic Visualiser로 수작업). 7개 기법(좌수 4개: vibrato/Point Note/Upward Portamento/Downward Portamento, 우수 3개: plucks/glissando/tremolo). 클래스 불균형이 매우 심함(plucks 11860개 vs tremolo 77개). 79/10/10곡으로 train/val/test 분리.

C. 결과
F1 86.54%(precision 87.62%, recall 85.48%)로, 같은 데이터셋에서 재현한 기존 모델들(GZFNO 68.73%, EHFCN 68.65%, CNN+Res 70.77%)을 모두 크게 앞섬. Ablation에서는 self-attention 제거 시 F1 2.53%p 하락, residual skip-connection 제거 시 1.65%p 하락, multi-scale을 single-scale로 바꾸면 F1이 9.56%p나 급락 — multi-scale 구조가 가장 핵심적인 기여. Confusion matrix에서는 데이터가 가장 많은 plucks가 다른 소수 기법들을 흡수하는 방향으로 오분류가 발생하고, Point Note는 vibrato와 혼동되기 쉬움(짧은 음에서는 음향적으로 거의 동일).

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: 같은 연구팀(4번 논문)의 후속 연구로, "합성 데이터로는 충분한 일반화를 보장할 수 없다"는 문제의식을 더 강하게 밀어붙여 이번엔 진짜 연속 연주 99곡을 직접 라벨링함 — 우리도 OrchideaSOL이 원래 고립음으로 설계된 데이터셋이라는 점에서 완전한 진짜 연주는 아니지만, 합성 콘캐트네이션보다는 중간 정도의 현실성을 가짐. 클래스 불균형(plucks 압도적 vs tremolo 희소) 문제와 그로 인한 소수 클래스 흡수 현상도 우리의 col_legno_battuto(227개, 가장 적음) 문제와 본질적으로 같은 현상.

차이점:
1. 과제 정의 — 이 논문은 한 프레임에 여러 기법이 동시에 존재할 수 있는 multi-label 문제이고, 우리는 한 클립에 정확히 하나의 기법만 있는 single-label 문제. 우리 문제가 훨씬 단순한 부분집합.
2. 데이터 현실성 — 이 논문은 실제 연주자가 연주한 연속 음악 99곡을 그대로 쓰고, 우리는 미리 설계된 고립음 샘플(OrchideaSOL)을 씀 — 이 논문이 더 현실적인 데이터를 다루지만, 우리는 노이즈가 적은 깨끗한 비교 실험이 가능함.
3. 모델 구조 — 이 논문은 HRNet 스타일 multi-scale + self-attention의 비교적 복잡한 구조. 우리는 v1(SVM/XGBoost/RF + 245차원 feature)과 v2(SmallCNN)로 상대적으로 단순한 구조.
4. 평가 단위 — 이 논문은 frame-level precision/recall/F1, 우리는 클립 단위 macro F1(LOSO) — 평가 철학이 다름.

적용 가능한 인사이트: 이 논문의 confusion matrix 분석(plucks가 소수 클래스를 흡수)은 우리 결과를 다시 살펴볼 때, col_legno_battuto가 구체적으로 어떤 클래스(ordinario나 pizzicato_secco 등)로 흡수되는지 짚어보고 같은 "다수 클래스 흡수" 패턴인지 확인해볼 근거가 됨. Multi-scale 구조가 길이가 다른 기법들을 동시에 잘 잡는다는 ablation 결과(single-scale로 바꾸면 9.56%p 급락)는, 우리 v2 CNN의 receptive field가 col_legno_battuto 같은 짧고 transient한 기법과 tremolo 같은 상대적으로 긴 기법을 모두 잘 잡고 있는지 점검해볼 동기가 됨.
