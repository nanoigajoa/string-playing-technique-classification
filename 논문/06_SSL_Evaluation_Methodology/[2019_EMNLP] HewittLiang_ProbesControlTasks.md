[2019_EMNLP] Designing and Interpreting Probes with Control Tasks

저자: John Hewitt, Percy Liang
출판: EMNLP-IJCNLP 2019 (Best Paper Runner-Up), ACL Anthology D19-1275
링크/DOI: https://aclanthology.org/D19-1275/
읽은 날:
상태: To Read   ← 아래 §1·2·D는 초록/지식 기반 스텁. §3·4는 정독 후 채울 것.


CORE

1. 한줄 요약
probe가 높은 정확도를 내는 것이 "표현이 그 정보를 담아서"인지 "probe 자신이 과제를 학습해서"인지
가르기 위해, 무작위 라벨을 맞추는 **control task**를 도입하고 그 둘의 격차(selectivity)로
probe의 신뢰도를 평가하자고 제안한다.

2. 핵심 내용 & 주요 기여
- probe 정확도 단독은 표현의 정보량을 과대평가할 수 있음을 지적.
- **control task**(구조가 없는 무작위 목표)를 같은 probe로 학습 → 둘의 차이(selectivity)가 커야
  "표현이 실제로 정보를 담는다"고 말할 수 있음.
- 선형/저용량 probe 선호 등 probe 설계 지침 제시.

3. 한계 / 비판점
  [정독 후] — NLP(단어 표현) 맥락. 회귀형 probe(우리 descriptor R²)에 selectivity 개념을
  어떻게 이식할지 확인.

4. 추가로 읽을 references
  [정독 후]


D. 내 프로젝트와의 비교 — **probe + 대조군의 이론적 정당화**
유사점: 우리는 descriptor probe에 **음성/chroma 대조군**을 이미 두고 있다
(`results/phase3/descriptor_probe.md`: chroma만 깊이에 따라 상승 → 대조군이 제 역할). 이 논문이
바로 "대조군 없는 probe 해석은 위험하다"는 방법론적 근거다.
적용 인사이트:
- Method의 probe 소절에서 "control을 두는 이유"를 이 논문으로 정당화 → 리뷰어의 "probe가
  스스로 학습한 것 아니냐" 공격 선제 차단.
- 우리 대조군 설계를 이들 프레임(selectivity)으로 재서술 가능: "저수준 물리량은 저층에서
  선형 가용(높은 R²) vs 화성 대조군은 깊이 의존" = 표현이 실제로 저수준 물리를 담는다는 증거.
- probe는 **저용량(선형/ridge)** 유지 권고와 우리 설계 일치 → "관행 준수" 근거로 인용.
예상 장애물: 이들의 selectivity는 분류(무작위 라벨)용. 우리는 회귀(R²)라 무작위 target의
control R²를 별도 정의해야 함 — 이식 시 "회귀형 control"을 명확히 기술할 것.
