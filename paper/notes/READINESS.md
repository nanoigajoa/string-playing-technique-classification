# 원고 검토 기록 — 2026-10-01

## 완료한 작업

- 국·영문 canonical v1에 관련 연구와 참고문헌 6편, 전처리·탐색 설정, 결론과 자료 접근 설명을 추가했다.
- 저장 예측에서 파생한 CSV로 성능 곡선, 재표집 민감도, 클래스 혼동 그림을 만들었다. `figures/manifest.json`에 입력·출력·생성 코드 SHA-256을 기록했다. PNG는 300 dpi이며 PDF 원본도 보관했다.
- 국·영문 DOCX를 갱신했다. 각 파일의 ZIP 무결성과 편집 가능한 표 2개·그림 3개를 확인했다.
- `review_preview.html`은 그림과 CSS를 내장한 영문 검토본이다. 같은 내용의 `review_preview.pdf`는 8쪽이며, 표지에는 제목·초록만 있다. 저자·소속은 기입하지 않았다. 전체 페이지를 이미지로 확인했다.
- 영문 초록은 공백 기준 188단어, 연구 기여와 재현 자료 설명을 보강한 원고 전체는 2,381단어다. 최종 제출본에서는 다시 집계한다.
- 수치 대조 스크립트에서 원고별 8행, README 4행 및 CNN 수치를 확인했다. 이 단계에서 추가 모델 학습은 수행하지 않았다.

## 형식 근거

[TISMIR 공식 제출 안내](https://transactions.ismir.net/about/submissions)를 2026-09-30 확인했다. 제출 형식은 PDF, 전체 길이는 참고문헌 포함 8,000단어 이내, 연구 논문 초록은 250단어 이내다. 익명 표지·번호가 붙은 절과 그림·저자/연도 인용을 적용했다. 템플릿 사용은 선택 사항이며 이번 검토본은 공식 템플릿을 적용한 파일이 아니다. 영문이 제출 언어이므로 국문은 내부 대조용이다.

## 참고문헌 확인 범위

| 문헌 | 확인한 일차 자료 | 원고에서 사용하는 범위 |
|---|---|---|
| Cella et al., 2020 | [Zenodo 원자료 기록](https://zenodo.org/records/3686252) | 데이터 출처·라이선스·권장 5분할 |
| Kruger & Jacobs, 2020 | [출판사 기록과 초록](https://www.tandfonline.com/doi/abs/10.1080/09298215.2020.1784957) | 특징 기반 현악 주법 분류의 선행 연구. 전문의 분할 방식은 확인하지 않았으므로 비교 주장에 사용하지 않음 |
| Li et al., 2024 | [ICLR 최종 논문 기록](https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) | MERT 배경과 서지 사항. 실제 전처리는 로컬 추출 코드에서 확인 |
| Oliver et al., 2018 | [NeurIPS 논문 기록](https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html) | 감독학습 대조군을 함께 보고할 근거 |
| Kapoor & Narayanan, 2023 | [저자 소속기관 논문 기록](https://collaborate.princeton.edu/en/publications/leakage-and-the-reproducibility-crisis-in-machine-learning-based-/) | 누수 문제의 연구 배경과 최종 서지 사항 |
| Rodríguez-Algarra et al., 2019 | [TISMIR 논문](https://transactions.ismir.net/articles/10.5334/tismir.24) | 평가 교란과 통제된 개입의 배경 |

이 문헌 목록은 현재 범위의 근거를 정리한 것이며 체계적 문헌 고찰은 아니다. 다른 과제·분할의 발표 수치와 직접 성능 비교를 하지 않았다.

## 남은 작업

1. **공개 등록:** 로컬 후보 ZIP과 검증을 완료했다. 코드 라이선스·최종 저자 선언·영구 보관 주소는 아직 확정하지 않았다. 자료 준비와 외부 게시를 구분한다.
2. **추가 연구 증거 검토:** 연구 기여 문장을 국·영문 원고에 반영하고 `docs/phase1_contribution_scope.md`에 증거와 한계를 정리했다. 자체 구현 결함 수정과 네 악기종의 관찰만으로 연구 논문의 충분한 기여가 확보됐다고 판단하지 않았다. 독립 자료나 결함별 통제 실험을 추가한다면 별도 사전 계획이 필요하다.
3. **최종 저자 정보와 선언:** 저자 역할·지원·이해상충·AI 도구 사용 설명은 확인된 정보로 작성해야 한다. 현재 원고에 임의의 선언을 넣지 않았다.
4. **최종 원고 편집:** 인용 전문 확인 범위를 넓히고 논문 기여에 필요한 문헌을 보완한다. 영문 표현, 그림·표의 읽기 크기, 익명 자료 접근, 제출 당시 규정을 최종 확인한다.

현재 PDF는 내부 검토본이다. 학술지 제출이나 자료 공개는 수행하지 않았다.

## 2026-10-01 재현 자료 준비

`../release/canonical-reproduction.zip`은 코드·분할·경로 제거 메타데이터·저장 예측·manifest·표·그림·원고를 포함한 로컬 공개 후보다. 원본 음원, mel/임베딩 캐시, checkpoint는 포함하지 않는다. 원래 manifest는 바이트를 유지하며, 변환된 메타데이터와 내보낸 명시적 분할은 별도 release manifest로 추적한다.

실제 ZIP을 새 폴더에 풀어 별도 Python 가상환경에서 저장 예측 100파일의 모델 예측 벡터 244개를 확인했다. 표 10개가 절대 허용 오차 1e-12 이내로 원본과 일치했고, 재계산 표에서 그림 3개를 생성했다. 이 환경에는 PyTorch가 없으며 checkpoint 추론·재학습을 수행하지 않았다. 자세한 실행·환경·묶음 SHA-256은 `../release/VALIDATION.md`와 `VALIDATION.json`에 있다.

## 재생성

저장소 루트에서 다음을 실행한다.

```bash
venv/bin/python -B src/phase1_canonical_figures.py
venv/bin/python -B src/phase1_manuscript_consistency.py
venv/bin/python -B src/phase1_manuscript_export.py
```

내보내기에는 로컬 pandoc이 필요하다. PDF는 생성된 HTML을 브라우저에서 A4로 인쇄하고 브라우저 머리글·바닥글을 끈다. HTML 안의 CSS가 여백과 표지 분리를 지정한다. PDF를 갱신하면 페이지 배치와 `artifact_manifest.json`도 다시 확인해야 한다.
