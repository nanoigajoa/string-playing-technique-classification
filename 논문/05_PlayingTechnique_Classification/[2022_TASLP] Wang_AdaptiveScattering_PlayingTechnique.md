[2022_TASLP] Adaptive Scattering Transforms for Playing Technique Recognition

저자: Changhong Wang, Emmanouil Benetos, Vincent Lostanlen, Elaine Chew
출판: IEEE/ACM Transactions on Audio, Speech and Language Processing, Vol. 30, pp. 1407-1421, 2022
링크/DOI: 10.1109/TASLP.2022.3156785
읽은 날: 2026-07-05
상태: Done


CORE

1. 한줄 요약
적응형 산란 변환(adaptive scattering)이라는 범용 프레임워크로 7가지 CBF 기법(PMT 4종 + PET 3종)을 MFCCs·MPS 대비 대폭 향상된 성능으로 인식하며 SOL 등 다중 데이터셋 검증.

2. 핵심 내용 & 주요 기여
adaptive scattering은 웨이블릿 변수 중 하나 이상에 데이터 기반 차원 축소를 포함하는 산란 변환이다. 두 가지 인스턴스를 도입한다: (1) 주파수 적응형 산란(AdaTS/AdaTRS) — PMT(vibrato, tremolo, trill, flutter-tongue) 탐지를 위해 지배 주파수 대역 주변을 적응적으로 분해, (2) 방향 적응형 JTFS(dJTFS-avg) — PET(acciaccatura, portamento, glissando) 탐지를 위해 피치 변화 방향(상향/하향)에 대해 평균화. CBFdataset을 새로 공개하며, 4개 데이터셋(CBFdataset, VPset, SOL, VocalSet)에서 SVM으로 검증해 MFCCs/MPS 대비 멀티클래스 macro F에서 큰 향상을 보인다. 이 논문은 2020 ICASSP jTFST 논문의 TASLP 확장판이다.

3. 한계 / 비판점
단음(모노포닉)만 처리하며, 코아티큘레이션 샘플을 제거해 실험을 단순화했다. 지배 주파수 대역 궤적이 불안정할 경우(옥타브 점프) AdaTS 성능이 저하된다. VPset에서 vibrato F-measure가 FDM 대비 낮다. 하이퍼파라미터(T, Q, M 값)를 수작업으로 기법 물리적 특성에서 도출해야 하므로 새 기법 추가 시 재조정이 필요하다.

4. 추가로 읽을 references
Andén & Mallat (2019), "Joint time-frequency scattering" — JTFS 이론 원논문
Wang et al. (2020) ICASSP — 본 논문의 컨퍼런스 선행 버전
Yang (2016), PhD dissertation — erhu portamento 전산 모델링 (VPset 기반)
Lostanlen et al. (2018) DLfM — SOL 데이터셋 기반 확장 기법 인식


OPTIONAL

A. 방법론
PMT 탐지: scalogram에서 지배 주파수 대역 궤적 λ_max(t) 추출 → L개 대역을 지배 대역 중심으로 분해 → AdaTS(시간 웨이블릿 적용, 변조율/형태 인코딩) → AdaTRS(AdaTS에 주파수 산란 추가, 변조율 축 스펙트럼 구조 캡처) → AdaTS+AdaTRS 연결 입력.

PET 탐지: JTFS S2^JTFS 계산(시간×주파수 동시 웨이블릿) → 방향 변수 θ=±1(피치 상향/하향) → dJTFS-avg: θ에 대해 평균화 → 방향 불변 표현.

분류기: SVM(Gaussian 커널, C∈2^{3:1:8}, γ∈2^{-12:1:-7} 그리드서치). CBFdataset은 8:2 연주자 분리, 5-fold CV.

B. 데이터셋
CBFdataset(공개): 10명 전문 연주자, 7기법(vibrato/tremolo/trill/flutter-tongue/acciaccatura/portamento/glissando), 전체 공연 4곡 + 이솔레이션 기법 녹음, 44.1kHz/24-bit. URL: c4dm.eecs.qmul.ac.uk/CBFdataset.html

추가 검증:
- VPset: erhu 4곡+바이올린, vibrato(25min)/portamento(55min)
- SOL v0.9HL: 12악기, 140기법, 9.8시간 이솔레이션
- VocalSet: 20명 가수, 10가지 성악 기법

C. 결과
CBFdataset (frame-based F-measure, 바이너리):
- PMTs — AdaTS+AdaTRS 평균 73.5% vs Standard scattering 72.8%
  (Flutter-tongue 89.2%, Trill 90.4%, Vibrato 72.0%, Tremolo 48.8%)
- PETs — dJTFS-avg 평균 75.7% vs JTFS 74.5%
  (Acciaccatura 74.8%, Portamento 66.0%, Glissando 86.8%)

CBFdataset (멀티클래스, 7기법): macro F=79.9% vs MFCCs 35.9%, MPS 52.0%

SOL (멀티클래스, 17기법): Scattering macro F=75.5% vs MFCCs 27.1%, MPS 26.6%

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: SOL 데이터셋(내 OrchideaSOL 전신)을 사용하며, SVM을 분류기로 사용하는 점이 동일하다. tremolo가 내 기법과 겹치고, 기법별 분류 난이도 패턴(tremolo가 가장 낮은 48.8%)이 내 프로젝트 결과와 비교 가능하다.

차이점:
1. 데이터셋 — 본 논문은 CBFdataset(관악기 중심)이 주 평가 데이터. SOL은 검증용 추가 데이터로만 사용.
2. 기법 종류 — 7가지 CBF 기법(PMT+PET) vs 내 5가지 현악기 기법(보우잉 중심).
3. 피처 — adaptive scattering(이론적으로 설계된 수작업 피처) vs 내 프로젝트의 mel-spectrogram 기반 학습.

적용 가능한 인사이트: SOL에서 17기법 macro F=75.5%라는 수치가 내 5기법 SVM 91.6%와의 직접 비교 기준점이 된다(클래스 수 차이 고려 필요). AdaTS/dJTFS 피처를 내 파이프라인에 추가 피처로 실험하면 특히 tremolo 탐지 향상에 도움이 될 수 있다.
