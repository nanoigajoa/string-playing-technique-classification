# 음악·MIR 선행연구 재조사와 v2 발전 방향

검토일: 2026-10-23. 대상: 국문 v2, 기존 주법 분류 독서 노트, 공식 학회 프로그램·저자 원문·출판사 본문. 새 학습이나 원고 수치 변경은 수행하지 않았다.

## 판단

최신 문헌에 연결해 질문을 좁히는 방향을 권한다. 학부 연구에서는 새 기반 모델 개발보다, 기존 방법으로 음악적으로 구체적인 질문 하나를 적절한 대조와 함께 해결하는 범위가 현실적이다. 기여의 크기와 실험 수는 같지 않다.

현 v2의 모델·라벨 예산·자기학습·재표집·감사 사례를 모두 중심 기여로 유지하면 초점이 흐려진다. 주법 인식에서 음의 시간 구간을 어떻게 사용할지에 집중하고, 기존 비교를 기준선으로 활용하는 것이 우선 후보다. 이는 아직 신규성이 확정된 연구계획이 아니다.

## 조사 범위와 한계

- 2024–2026 공개 연구를 우선 검색하고 직접적인 계보인 2019–2022 연구까지 역추적했다.
- 주요 검색: playing technique / violin / bowed strings / temporal context / onset / timbre similarity / pretrained representation, ISMIR·TISMIR·JNMR 및 음악·오디오 연구 출판 경로.
- 음악 학회·저널을 투고 후보로 삼았다. ICASSP는 음악 전용 학회가 아니지만 MERTech와 VioPTT는 직접 선행연구이므로 보조 문헌으로 포함했다. 이들을 ISMIR 논문으로 표시하지 않았다.
- 검색 결과의 crawl/published 상대 날짜는 출판 연도로 쓰지 않았다. 본문·서지·공식 프로그램의 연도를 기준으로 했다.
- 접근한 핵심 원문의 방법·데이터·평가 부분을 확인했다. 모든 검색 후보를 전문 정독한 체계적 문헌고찰은 아니다. 검색에서 못 찾았다는 사실은 해당 연구가 없다는 증거가 아니다.
- 2026 ISMIR는 11월 개최 예정이다. 프로그램에 실린 연구를 이미 발표가 끝난 논문으로 표시하지 않았다.
- TISMIR 기존 사이트의 일부 안내와 KEAMS 홈페이지는 도구에서 접근 오류가 발생했다. TISMIR 논문은 출판 플랫폼 본문, 에밀레는 학회 명의 공고와 게재 기록을 보완 근거로 사용했다. 확인하지 못한 향후 마감·비용·색인 상태는 단정하지 않는다.

## 우선 문헌과 현재 연구에 주는 의미

### 1. Ducher & Esling (2019), ISMIR

**Folded CQT RCNN for Real-time Recognition of Instrument Playing Techniques.**
[학회 원문](https://archives.ismir.net/ismir2019/paper/000086.pdf)

확인: 본문 §3–5, 특히 §5.3. 첼로 주법을 여러 속성으로 나누고 사운드뱅크 보류 및 시간에 따른 인식 반응을 평가한다. 주법 속성별로 필요한 문맥이 다르다는 논의가 이미 있다.

연결: 시간 구간 연구의 가장 직접적인 출발점이다. “주법마다 시간이 다르게 필요하다” 자체를 새 발견으로 내세우면 안 된다. 현대 고정 표현·음향 특징, 네 현악기종 보류, 같은 길이의 초반/후반 구간이라는 구체적 확장으로 차별점을 검토한다. 사운드뱅크 보류와 악기종 보류는 같은 일반화 단위가 아니다.

### 2. Kruger & Jacobs (2020), Journal of New Music Research

**Playing technique classification for bowed string instruments from raw audio.**
[출판사 기록](https://www.tandfonline.com/doi/abs/10.1080/09298215.2020.1784957)

확인: 이번에는 공식 서지·초록, 로컬 기존 검토 기록. 특징과 SVM으로 바이올린족의 주법을 분류하는 직접 선행연구다.

연결: 현재 245차원 특징은 이 연구의 정확한 재현이 아니다. 자체 기준선임을 유지해야 한다. JNMR가 이 과제를 다룬다는 근거지만, 과거 게재가 현재 원고의 게재 충분성을 보장하지 않는다.

### 3. Li et al. (2022), ISMIR

**Playing Technique Detection by Fusing Note Onset Information in Guzheng Performance.**
[학회 원문](https://archives.ismir.net/ismir2022/paper/000037.pdf)

확인: 원문의 문제 설정·데이터·온셋 결합 방식. 고쟁의 프레임별 주법 예측을 음의 시작점 정보와 결합하여 음표 단위로 집계한다.

연결: 음표 경계를 이용하는 것 자체는 신규성이 아니다. 현재 데이터에서는 이미 나뉜 클립의 시작을 정확한 실제 발음 시작과 혼동하지 말아야 한다. 온셋 정렬을 쓴다면 주석 또는 검출 오차를 별도로 점검해야 한다.

### 4. Zhang, Liang & Dixon (2024), ISMIR

**From Audio Encoders to Piano Judges: Benchmarking Performance Understanding for Solo Piano.**
[저자 원문](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2024/ZhangEtAl-ISMIR-2024.pdf)

확인: 초록, 연구 질문, 인코더와 PLD 과제 설계. 사전학습 표현을 피아노 연주 수준·난이도·기법이라는 구체적인 음악 과제로 평가한다.

연결: 표현 비교도 음악적 연구가 될 수 있다는 선례다. 다만 자체 데이터와 음악 과제 정의의 기여가 있으므로, 단순 모델 비교만으로 충분하다는 근거로 쓰면 안 된다. 우리 연구에서는 “고정 표현이 어떤 발음 구간에서 주법 정보를 제공하는가”로 질문을 구체화할 수 있다.

### 5. Tian, Lattner & Saitis (2025), ISMIR

**Assessing the Alignment of Audio Representations With Timbre Similarity Ratings.**
[학회 게재본 기록](https://zenodo.org/records/17706569) · [프로그램](https://ismir2025program.ismir.net/poster_283.html) · [저자 사전 원고](https://arxiv.org/abs/2507.07764)

확인: 공식 초록·게재본 기록 및 공개 심사평. 사람의 음색 유사도 판단에 표현 간 거리가 얼마나 부합하는지 평가한다. 게재본 설명에서는 MFCC도 경쟁력이 있음을 보고한다. 초기 초록의 강한 우위 표현과 게재본의 완화된 표현을 혼용하지 않는다.

연결: 높은 분류 점수와 지각적으로 타당한 음색 표현은 다른 질문이다. 현 v2의 혼동행렬만으로 인간이 느끼는 주법 유사성을 설명할 수 없다. 실제 청취 판단 자료를 모으는 후속 연구라면 이 계보가 적합하다. 현재 1차 확장에서는 별도 인간 실험까지 동시에 벌이지 않는 편을 권한다.

### 6. Papaioannou, Benetos & Potamianos (2025), ISMIR

**Universal Music Representations? Evaluating Foundation Models on World Music Corpora.**
[공식 학회 목록](https://ismir.net/conferences/ismir-2025/) · [저자 원문](https://arxiv.org/html/2506.17055v1) · [코드](https://github.com/pxaris/FM-music-tagging)

확인: 게재 사실, 원문 연구 범위·평가 설정. 여러 문화권의 음악 자료에서 사전학습 표현과 전이 전략을 평가한다.

연결: 사전학습의 유용성을 조건부로 묻는 것은 현행 MIR의 연구 질문이다. 하지만 서로 다른 코퍼스의 성능 차이만으로 문화적 원인을 분리할 수는 없다. 현악기 연구를 문화권 비교로 확대하는 것은 이번 범위를 넘어선다.

### 7. Zhou et al. (2025), TISMIR

**CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research.**
[출판 플랫폼 본문](https://reference-global.com/article/10.5334/tismir.194?tab=article)

확인: §3의 데이터 구성·전처리·평가 준비. GZ IsoTech, Guzheng Tech99, Erhu Playing Technique 등을 통합해 접근성을 개선한다.

연결: 후속 독립 데이터 후보를 찾는 출발점이다. 공개되어 있다는 이유만으로 현재 다섯 현악 주법과 라벨이 호환되는 것은 아니다. 클립 생성 이전 원녹음·연주자·사운드뱅크 ID와 라벨 의미를 확인해야 한다. 원논문의 분할을 무조건 답습하지 않고 우리 질문에 맞는 그룹 경계를 정해야 한다.

### 8. Stefani & Turchet (2025), EURASIP Journal on Audio, Speech, and Music Processing

**Real-time playing technique recognition embedded in a smart acoustic guitar.**
[출판사 전문](https://link.springer.com/article/10.1186/s13636-025-00413-6)

확인: §3–4의 입력 구간·지연·연주자/악기 쌍 보류 설계. 여덟 기타 주법을 임베디드 시스템에서 인식하며 짧은 문맥과 일반화를 평가한다.

연결: 시간 구간과 성능을 연결하는 최신 직접 선례다. 우리의 악기종 보류는 새 연주자/개체 검증이 아니다. 오프라인 크롭 길이만 측정하면 실시간 지연을 검증한 것으로 부를 수 없다. 이 저널은 음악 전용은 아니지만 음악 신호처리를 직접 다룬다.

### 9. Li et al. (2024), ICASSP — 보조 직접 선행연구

**MERTech: Instrument Playing Technique Detection Using Self-Supervised Pretrained Model With Multi-Task Finetuning.**
[저자 원고](https://arxiv.org/abs/2310.09853) · [기관 보관 게재 승인본](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/93901/Benetos%20MERTech%3A%20instrument%20playing%20technique%20detection%20using%20self-supervised%20pretrained%20model%20with%20multi-task%20finetuning%202024%20Accepted.pdf?isAllowed=y&sequence=2)

확인: 초록·방법의 주법/음고/온셋 결합과 발표 기록. 사전학습 모델을 주법 인식에 적용하는 신규성은 이미 선점되어 있다. 현재 MERT 마지막 층 평균 + 헤드는 MERTech 전체 방법의 재현이 아니다.

### 10. Wang et al. (2026), ICASSP — 보조 직접 선행연구

**VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation.**
[2026-02-03 v3 원문](https://arxiv.org/html/2509.23759v3) · [프로젝트](https://y10ab1.github.io/VioPTT/) · [저자 출판 목록](https://homepage.iis.sinica.edu.tw/pages/lisu/publications_en.html)

확인: v1과 v3의 서술, 방법·자료·평가, ICASSP 2026 발표 서지. 합성 바이올린 자료와 전사 정보를 주법 분류에 연결한다. 주법 평가의 RWC 고립음과 전사 평가의 URMP/Bach10을 구분해야 한다.

연결: 음의 시간 전개를 쓰는 방향에 동기를 준다. 다만 현재 다섯 클래스와 겹침이 제한되고, 합성 자료 전체를 재현하는 비용도 별개다. 이 논문의 MERTech 비교 점수를 MERT 일반의 바이올린 한계로 옮겨 쓰면 안 된다. 논문 자체는 현재 2025 프리프린트로만 분류하면 부정확하다.

### 11. Wang et al. (2026), ISMIR — 발표 예정, 내용 미확인

**VioTech: Frame-Level Violin Playing Technique Detection with Multi-Scale Mixture-of-Experts.**
[공식 프로그램](https://ismir2026program.ismir.net/papers.html?session=10) · [저자 기관 출판 목록](https://web.iis.sinica.edu.tw/en/page/ResearchOverview/Publications/2026.html)

확인: 제목·저자·ISMIR 2026 프로그램 등재. 이번에는 상세 원문과 수치·데이터·코드를 확보하지 못했다. 제목에서 암시되는 구조 외의 세부 성능·약점은 추측하지 않는다. 시간 구간/다중 스케일 접근의 신규성을 최종 주장하기 전에 가장 먼저 대조할 최신 후보다.

### 추가 발견, 우선순위 낮음

- [GOAT, ISMIR 2025](https://ismir2025program.ismir.net/poster_245.html): 기타 오디오와 주법 표현이 가능한 타브 자료. 현악기종 다섯 주법의 직접 외부 테스트로 쓰기는 어렵다.
- [Layer-wise Investigation…, 2025](https://arxiv.org/abs/2505.16306): MusicFM/MuQ 층별 분석. 이번 확인에서는 음악 학회 게재를 확정하지 않았다. 층별 probe만 추가하는 방향은 기존 연구와 겹친다.
- [Dimension-Wise Probing…, Electronics 2026](https://www.mdpi.com/2079-9292/15/15/3421): 음악 전문지 범위 밖이므로 투고 선례로 추천하지 않는다. 다만 고정 표현·주법·출처별 분할이 겹치는 경쟁 문헌 후보로 확인했다. 이 논문의 주장 정확성 전체를 감사하지 않았다.

## 기존 독서 노트에서 재검토할 부분

현재 저장소에는 직접 선행연구가 이미 상당수 있다. 따라서 “아무 선행조사도 하지 않았다”보다 “문헌을 설계와 주장 경계로 연결하는 과정이 약했다”가 더 정확하다.

VioPTT 노트에는 77.22% 조건을 `최고(full ablation)`이라고 적었으나 프로젝트 표는 모든 전사 특징을 쓰는 조건을 `No ablation`으로 표시한다. 또한 그 결과를 SSL의 악기 전이 한계 전반으로 확대하는 문장은 좁혀야 한다. Ducher 노트의 분류 축을 이용해 우리 col legno 오류 원인을 설명하는 부분도 검증된 원인이 아니라 가설이다. 이번에는 역사적 노트 자체를 수정하지 않았다.

## 권장 질문: 음의 어느 구간이 주법 정보를 제공하는가?

잠정 제목: **현악 주법 인식에서 발음 구간의 역할: 음향 특징과 고정 음악 표현의 악기종 간 전이 평가**.

핵심 질문: 발음 초반과 이후의 동일 길이 구간에서 얻는 주법 분류 성능은 어떻게 다르며, 그 양상이 특징 기반 표현과 고정 음악 표현 및 보류 악기종에서 유지되는가?

현재 다섯 클래스를 유지하되 해석은 세 대비에 집중한다.

- pizzicato secco / col legno battuto: 뜯기와 타격을 구분하는 초반 구간의 역할.
- ordinario / sul ponticello: 접촉 위치에 따른 지속 구간의 역할.
- ordinario / tremolo: 반복 활긋기를 관찰하는 구간 길이의 역할.

위 문장은 검증할 가설이다. 반드시 예상 방향의 결과가 나와야 하는 것은 아니다. 기존 결과를 보고 선택한 탐색이라는 사실도 유지한다.

### 학부 규모의 최소 설계

1. 파일명·클립 경계·음량·길이·실제 발음 시작을 먼저 점검한다. 클래스와 악기별로 동일 규칙의 크롭이 가능한지 확인한다.
2. 기존 다섯 클래스와 악기종 보류를 유지한다. 주 실험의 라벨 예산은 하나로 고정하고, 필요할 때만 낮은 예산 하나를 보조로 둔다. 네 예산 × 모든 모델 × 모든 전처리를 처음부터 확장하지 않는다.
3. 전체 클립 기준선, 같은 길이의 초반 구간, 이후 구간을 비교한다. 길이 효과와 위치 효과를 분리한다. 정확한 구간은 학습/개발 자료로 확정한다.
4. 음향 특징과 고정 MERT를 공통 분류기로 비교한다. 분류기 민감도 대조를 작게 추가할 수 있다. 마지막 층만의 결과를 표현 전체의 한계로 부르지 않는다.
5. 구간 결합이나 시간별 pooling을 시험한다면 같은 입력 시간·출력 차원·분류기 용량을 맞춘 대조가 필요하다. 개선이 나오면 단순 구성의 효용을, 없으면 구간별 정보의 차이를 보고한다.
6. macro F1, 클래스별 F1, 악기종별 결과와 대응 혼동 변화를 보고한다. 반복 시드가 독립 악기나 연주자의 수를 늘리지는 않는다.

### 타당성에 특히 필요한 사항

- 잘린 구간에 대한 MERT 입력은 원음부터 잘라 다시 인코딩한다. 전체 음원을 문맥화한 hidden state의 일부만 뽑으면 나중 구간 정보가 남아 있을 수 있다.
- “그 구간에 정보가 있는가”는 조건별 재학습으로, “전체 클립 모델이 그 구간 제거에 민감한가”는 고정 모델 교란으로 묻는 다른 질문이다. 둘을 섞지 않는다.
- 자연스러운 감쇠와 무음·패딩 비율, 전체 길이의 대리 단서를 분리한다. 이후 구간에 소리가 남은 표본만 택하면 클래스별 선택 편향이 생길 수 있다.
- 크롭 위치·창 길이·정규화·분류기 선택은 테스트 점수로 고르지 않는다. 후속 실험은 이미 탐색한 데이터에 대한 확장이지 새 독립 확인 실험이 아니다.
- 다른 악기종에서 유지된다고 새 연주자·새 녹음환경에서도 유지된다고 말하지 않는다.
- 오프라인 입력 구간 길이를 실제 시스템 지연이나 인간의 인식 시간으로 해석하지 않는다.

### 예상 기여와 중단 기준

성립할 수 있는 기여는 현재 데이터의 악기종 전이 조건에서 구간별 주법 정보와 표현 선택의 상호작용을 설명하는 재현 가능한 평가다. 시간 문맥의 중요성 자체, 새 기반 모델, 보편적인 주법 음향 법칙이 아니다.

기존 2019/2025 연구의 일반적인 결과만 반복하고 새 조건에서도 별도의 설명이나 활용 지침을 얻지 못하면 국제 전문지 기여가 크게 강화됐다고 평가하지 않는다. 그때는 제한된 재현·확장 보고로 마무리하거나 더 작은 발표 형식을 선택한다. 모델 추가를 반복해 긍정적 결과를 찾는 방식으로 계속 확장하지 않는다.

## 다른 방향과 우선순위

| 방향 | 장점 | 부담/한계 | 순위 |
|---|---|---|---|
| 발음 구간·문맥과 악기종 전이 | 기존 음원·코드를 활용, 음악 질문이 명확 | 선행연구 중복과 크롭 교란 통제 필요 | 1 |
| 음고·강약을 맞춘 ordinario/sul ponticello 비교 | 더 좁은 음색 질문 | 공통 음역·강약의 표본이 충분한지 먼저 확인해야 함 | 2 |
| 사람의 주법 유사도와 표현 거리 | 음악 지각 연구와 직접 연결 | 청취 실험·설계·참가자 확보가 별도 과제 | 협력자가 있을 때 |
| 연속 연주의 주법 전사 | 최신 VioPTT/VioTech와 직접 연결 | 새 주석·자료·평가 단위가 필요, 현재 v2보다 큰 과제 | 이번 확장에서는 보류 |
| MERT/MuQ/MusicFM 등의 대규모 순위 비교 | 구현이 익숙함 | 구체적 음악 질문과 신규성 강화가 자동으로 되지 않음 | 비권장 |

## 음악 관련 투고 경로

아래는 범위 적합성과 연구 규모에 대한 판단이지 합격 가능성이나 학술지 서열표가 아니다. 학부생이라는 이유로 정규 심사의 기준이 완화된다고 가정하지 않는다.

| 후보 | 근거 | 현 단계 권고 |
|---|---|---|
| Sound and Music Computing (SMC) | [2026 공식 CFP](https://smc26.mbz.hr/en/submissions/call-for-papers)에 음악 인식·분류·연주 분석·MIR 명시 | 질문 하나와 통제된 실험을 완성한 뒤 차기 공고를 검토할 우선 국제 학회 후보. 2026 접수는 종료 |
| ISMIR 정규 논문 | [공식 CFP](https://ismir2026.ismir.net/authors/call-for-papers): 음악 분석·평가·재현성 포함 | 발전된 실험의 음악적 발견과 차별성이 충분할 때 도전. 지금 v2의 즉시 제출 권고는 아님 |
| ISMIR LBD | [공식 안내](https://ismir2026.ismir.net/call-for-late-breaking-demo): 초기 결과·신규 참여자 피드백 | 초기 결과 공유에 적합. 정규 학회 논문 및 공식 proceedings 게재와 구별. 2026 접수 종료 |
| Journal of New Music Research | [범위](https://www.tandfonline.com/journals/nnmr20/about-this-journal), 직접 현악 주법 논문 선례 | 좁고 음악적으로 해석 가능한 결과로 발전하면 저널 후보. 자동으로 쉬운 투고처는 아님 |
| TISMIR | CCMusic 및 TENT 등의 음악 데이터·주법 연구 선례 | 초기 학부 프로젝트의 기본 목표보다는 강한 평가/자료 기여를 갖춘 후의 확장 목표 |
| 에밀레, 한국전자음악협회 학술지 | [학회 명의 2026 공고](https://groups.google.com/g/cec-conference/c/mQv0F53lKvE), [2025 게재 사례](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003290862) | 국내 음악·컴퓨터음악 저널 후보. 원고가 독자에게 주는 음악 기술적 의미와 편집 범위를 먼저 맞출 필요. 2026 공고 마감 9/30은 경과 |
| 한국음악지각인지학회 학술발표 | [학회 세미나 안내](https://www.ksmpc.kr/contact) | 지각/연주 해석을 중심으로 발전할 때 피드백 경로. 순수 분류 성능 비교만으로 지각 연구를 했다고 포장하지 않음 |

2027 학회 일정·마감과 다음 에밀레 모집은 이번 조사에서 확인하지 않았다. 음악교육 효과를 실제로 평가하지 않은 상태에서 음악교육 전문지에 맞추려고 효용을 확대하지 않는다. 일반 ML 응용 저널을 이번 투고 권고의 중심에서 제외한다.

## 먼저 읽고 설명할 순서

1. Ducher & Esling 2019 §3, §5.3: 우리의 시간 구간 질문 중 이미 알려진 부분.
2. Stefani & Turchet 2025 §3–4: 문맥 길이와 일반화 평가를 분리하는 방법.
3. VioPTT 2026 §3–5: 전사 정보의 역할과 실제 주법 평가 자료의 범위.
4. Tian et al. 2025: 분류와 지각적 음색 타당성의 차이.
5. VioTech 2026: 전문을 확보한 뒤 제안한 확장과의 중복을 최종 점검.

각 논문에 대해 ‘무엇을 입력하고 무엇을 예측했는가 / 어떤 단위를 보류했는가 / 어떤 대조가 결론을 지지하는가 / 내 연구에서는 무엇을 다르게 검증할 것인가’를 한 문단으로 설명하는 것이 목표다. 참고문헌 수를 늘리는 것보다 이 연결이 우선이다.
