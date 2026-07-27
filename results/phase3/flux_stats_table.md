# Phase 3 보완 — Onset-strength envelope(코드명 flux) 클래스별 기술통계

연구과정/03_피드백_전문가.md 2순위: SHAP flux_std 방향성을 실측 중앙값으로 검증 (초기 가설 '낮음→ordinario, 높음→col_legno'은 실측에서 기각 — 아래 핵심 발견 참조).

## flux_mean_0

Kruskal-Wallis H=2731.93, **p < 0.001**

| 클래스 | n | mean | std | median | Q25 | Q75 |
|---|---|---|---|---|---|---|
| ordinario | 1193 | 0.7769 | 0.2141 | 0.8333 | 0.6465 | 0.9211 |
| pizzicato_secco | 639 | 0.1651 | 0.0926 | 0.1427 | 0.1050 | 0.1984 |
| tremolo | 1041 | 1.2471 | 0.2625 | 1.2103 | 1.0921 | 1.3635 |
| sul_ponticello | 302 | 0.7808 | 0.1373 | 0.8334 | 0.6905 | 0.8826 |
| col_legno_battuto | 227 | 0.1578 | 0.0886 | 0.1663 | 0.0715 | 0.2258 |

## flux_std_0 ★ SHAP per-feature 1위

Kruskal-Wallis H=2323.20, **p < 0.001**

| 클래스 | n | mean | std | median | Q25 | Q75 |
|---|---|---|---|---|---|---|
| ordinario | 1193 | 0.5101 | 0.0799 | 0.5115 | 0.4663 | 0.5594 |
| pizzicato_secco | 639 | 0.3865 | 0.3198 | 0.2706 | 0.1788 | 0.4883 |
| tremolo | 1041 | 1.0235 | 0.3582 | 0.9253 | 0.8052 | 1.0936 |
| sul_ponticello | 302 | 0.4989 | 0.0905 | 0.4802 | 0.4412 | 0.5373 |
| col_legno_battuto | 227 | 0.1837 | 0.0760 | 0.1804 | 0.1443 | 0.2060 |

## flux_q25_0

Kruskal-Wallis H=1898.75, **p < 0.001**

| 클래스 | n | mean | std | median | Q25 | Q75 |
|---|---|---|---|---|---|---|
| ordinario | 1193 | 0.4829 | 0.2229 | 0.5584 | 0.3313 | 0.6415 |
| pizzicato_secco | 639 | 0.0059 | 0.0129 | 0.0000 | 0.0000 | 0.0070 |
| tremolo | 1041 | 0.4720 | 0.1907 | 0.4871 | 0.3459 | 0.6215 |
| sul_ponticello | 302 | 0.5261 | 0.1744 | 0.5961 | 0.4019 | 0.6604 |
| col_legno_battuto | 227 | 0.0219 | 0.0279 | 0.0099 | 0.0007 | 0.0321 |

## flux_q75_0

Kruskal-Wallis H=2897.37, **p < 0.001**

| 클래스 | n | mean | std | median | Q25 | Q75 |
|---|---|---|---|---|---|---|
| ordinario | 1193 | 1.0283 | 0.2645 | 1.0753 | 0.8776 | 1.1980 |
| pizzicato_secco | 639 | 0.1627 | 0.0845 | 0.1470 | 0.1011 | 0.2145 |
| tremolo | 1041 | 1.8603 | 0.4942 | 1.7311 | 1.5544 | 1.9972 |
| sul_ponticello | 302 | 0.9723 | 0.1500 | 1.0140 | 0.8877 | 1.0838 |
| col_legno_battuto | 227 | 0.2409 | 0.1557 | 0.2703 | 0.0787 | 0.3734 |

## 핵심 발견 — flux_std_0 방향성 수치화

- col_legno_battuto median: **0.1804** (최저)
- ordinario median: **0.5115** (중간)
- 배율: ordinario가 col_legno보다 2.8× 높음

> SHAP 방향성: **낮은 flux_std → col_legno_battuto**, **중간 flux_std → ordinario**, **높은 flux_std → tremolo**
> 실측 중앙값(col_legno 0.18, ordinario 0.51, tremolo 0.93)이 이 순서와 일치한다.

## 출력 파일

| 파일 | 설명 |
|---|---|
| `flux_std_boxplot.png` | flux 4개 컬럼 클래스별 박스플롯 |
| `flux_stats_table.md` | 이 파일 — 기술통계 테이블 |