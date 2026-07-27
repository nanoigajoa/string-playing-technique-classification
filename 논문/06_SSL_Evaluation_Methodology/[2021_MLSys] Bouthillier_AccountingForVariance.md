[2021_MLSys] Accounting for Variance in Machine Learning Benchmarks

저자: Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, N. Mohammadi Sepahvand, Edward Raff, Kanika Madan, Vikram Voleti, Samira Ebrahimi Kahou, Vincent Michalski, Tal Arbel, Chris Pal, Gaël Varoquaux, Pascal Vincent
출판: MLSys 2021 (arXiv:2103.03098)
링크/DOI: https://arxiv.org/abs/2103.03098
읽은 날:
상태: To Read   ← 아래 §1·2·D는 초록/지식 기반 스텁. §3·4는 정독 후 채울 것.


CORE

1. 한줄 요약
모델 비교의 성능차가 "진짜 차이"인지 판정하려면 여러 분산 원천(데이터 샘플링, 초기화,
하이퍼파라미터, 데이터 증강)을 모두 모델링해야 하며, 일부 원천만 고정한 채 비교하면
없는 차이를 유의하게 만든다는 것을 정량적으로 보인다.

2. 핵심 내용 & 주요 기여
- 벤치마크 결과의 변동을 여러 독립 분산 성분으로 분해하고, 각 성분의 크기를 실측.
- (반직관) 분산 원천을 더 많이 무작위화한 불완전 추정량이, 하나만 반복하는 추정량보다
  이상적 추정량에 더 가깝고 계산비도 크게 절감(≈51×).
- 올바른 비교를 위한 통계 절차(성분을 포함한 신뢰구간/검정) 제안.

3. 한계 / 비판점
  [정독 후] — 우리 소규모(n=4 fold) 세팅에 이들의 분산분해가 그대로 적용되는지, 표본이 큰
  벤치마크 전제와의 간극을 확인할 것.

4. 추가로 읽을 references
  [정독 후]


D. 내 프로젝트와의 비교 — **함정③(재표집 단위)의 정면 인용처**
유사점: "분산 원천을 잘못 잡으면 CI가 좁아져 없는 차이를 유의하게 만든다"는 우리 B6 발견
(`results/phase1/cluster_bootstrap.md`)과 동일한 논지. 우리는 fold(악기)간 분산을 빼고
클립만 재표집한 1수준 CI가 5%·20%에서 SVM 우위를 "유의"하게 만들었고, fold를 포함한
2수준 CI에선 전 구간 0을 포함함을 보였다 → 이 논문이 그 현상의 일반 이론이자 인용 근거.
적용 인사이트:
- **Q2(수식 약점) 해결**: 총분산 = σ²_between-fold + σ²_within-clip 분해를 이들 프레임으로
  형식화 → 함정③을 손짓이 아닌 한 줄 식으로 승격. Method의 "uncertainty quantification" 소절 근거.
- 우리의 "지배적 분산 성분은 fold(악기)간"이라는 주장은 이 논문의 "어느 성분이 큰가를 실측하라"
  권고의 특수 사례로 배치 가능.
예상 장애물: 이들은 대규모 반복 전제, 우리는 n=4. "성분 크기 실측"은 못 해도 "빠진 성분이
지배적임을 논증"하는 수준으로 제한해 인용할 것(과대 적용 금지).
