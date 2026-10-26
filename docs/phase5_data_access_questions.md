# 후속 연구 데이터 접근·녹음 feasibility 검토 문안

2026-09-08. 검토용 초안이며 어떤 수신자에게도 발송하지 않았다. 소속·연구책임자·공개 범위는 실제 상황으로 채워야 한다.

**교수님/학과 내부 검토용**

현악 주법 인식 모델이 녹음 환경 변화에도 안정적인지, 새 연주자에서도 주법 차이를 구별하는지 확인하는 연구를 준비하고 있습니다. 바이올린·비올라·첼로·더블베이스 각 1명으로 pilot을 시작할 수 있는지 확인하고 싶습니다.

한 사람당 두 세션, 각 세션은 5주법 × 3음 × 2강약 × 2takes로 계획하고 있습니다. 실제 소요시간은 pilot에서 확인하며 우선 회당 45–75분을 가정합니다. 기본 주법은 ordinario, pizzicato secco, bow tremolo, sul ponticello, col legno battuto이고, 실행 지시는 현악 전공자의 검토를 받고자 합니다.

확인할 항목:

- 악기별 참여 가능 인원과 같은 연주자의 두 세션 참여 가능 여부
- col legno battuto 실행 및 악기·활 보호 조건
- 녹음 공간, 마이크·오디오 인터페이스, 녹음 담당자 사용 가능 여부
- 회당 보상·교통비·대관료와 예산 상한
- 연구용 내부 분석, 전문가 청취, 논문 예시, 공개 dataset 중 허용 가능한 범위
- pilot 이후 종별 2명 이상의 별도 confirmation 참여자를 추가 모집할 경로
- 연구참여 동의 및 소속 기관에서 필요한 절차의 담당자

**Violin Technique Dataset: metadata-first inquiry draft**

Source: https://zenodo.org/records/13773807

Subject: Metadata and research-access inquiry for the Violin Technique Dataset

Dear Dr. Tamer,

I am preparing a study of pretrained audio representations for bowed-string playing-technique recognition. Before requesting the full audio collection, I would like to check whether the Violin Technique Dataset fits our evaluation design.

Could you share the technique vocabulary and an anonymized count table by performer and recording session? We are particularly interested in ordinario, pizzicato secco, bow tremolo, sul ponticello, and col legno battuto. We would also like to know whether these labels are score-derived, manually verified from audio, or inferred by a model, and whether repeated sessions or microphone channels are identifiable.

Our planned analysis includes feature extraction, controlled mild audio transformations, and expert listening. Could you clarify the access procedure and permitted uses, separately for internal analysis, transformed excerpts used in listening evaluations, and publication of audio examples? A small metadata sample would be sufficient for the initial feasibility check.

[Name, affiliation, supervisor/contact]

**mdw cello project: subset inquiry draft**

Verified contact source: https://iwk.mdw.ac.at/pamies-vila/

Subject: Research-data feasibility inquiry: cello playing techniques and recording conditions

Dear Dr. Pàmies-Vilà,

I am preparing a study on whether audio representations remain stable under recording changes while retaining sensitivity to actual playing-technique changes. Your project on cello sound production beyond ordinary bowing appears particularly relevant.

Would any human-performed audio subset be available for research collaboration, together with anonymized performer/session identifiers and technique descriptions? Our current labels are ordinario, pizzicato secco, bow tremolo, sul ponticello, and col legno battuto. We distinguish battuto from tratto and do not assume that all these techniques are present in your collection.

For an initial feasibility check, a technique-by-performer count table, the available recording conditions, and the applicable use restrictions would be enough. We would be interested in internal feature analysis, controlled mild transformations, and expert listening; audio redistribution is a separate question and would require your explicit terms.

[Name, affiliation, supervisor/contact]

**TU-Note/RWC-I 확인에 사용할 공통 질문**

- 현재 연구용 다운로드/기관 내 이용 경로가 유지되는가?
- 각 파일의 마이크, 연주자, 날짜·세션, take를 연결하는 metadata가 있는가?
- TU-Note: 같은 번호 DPA/BuK가 같은 연주의 동시 녹음인가? 정렬·자르기 차이가 있는가?
- RWC-I: variation 1–3은 물리적 악기, 연주자, 녹음 조건 중 무엇의 구분인가?
- 다섯 주법별 실제 파일 수와 누락 조건은 무엇인가?
- 특징 추출·mild reverb/EQ·전문가 청취·논문 음원 예시의 허용 범위는 무엇인가?

답변이 오면 `확인 사실 / 불명확 / 불충족`으로 기록한다. 연락처만 확보한 상태를 데이터 확보로 표시하지 않는다.
