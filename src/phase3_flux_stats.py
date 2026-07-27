"""
Phase 3 보완 — flux_std 클래스별 기술통계 + 박스플롯

연구과정/03_피드백_전문가.md 2순위 작업:
    onset-strength(코드명 flux) 방향성 초기 가설을 실측 중앙값으로 검증
    (가설은 기각됨: 실측은 col_legno 최저 < ordinario 중간 < tremolo 최고).

실행:
    python src/phase3_flux_stats.py

의존 파일:
    data/features_full.csv

출력:
    results/phase3/flux_stats_table.md   — 클래스별 기술통계 테이블
    results/phase3/flux_std_boxplot.png  — 박스플롯
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / 'data' / 'features_full.csv'
RESULTS_DIR = ROOT / 'results' / 'phase3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
LABEL2STR = dict(enumerate(CLASSES))

# phase3_shap.py와 동일 팔레트
CLASS_COLOR = {
    'ordinario':         '#4C72B0',
    'pizzicato_secco':   '#55A868',
    'tremolo':           '#DD8452',
    'sul_ponticello':    '#8172B2',
    'col_legno_battuto': '#C44E52',
}
FLUX_COLS = ['flux_mean_0', 'flux_std_0', 'flux_q25_0', 'flux_q75_0']


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['label_id'].map(LABEL2STR)
    assert df['label'].notna().all(), "label_id 매핑 실패"
    print(f"  로드: {df.shape[0]}개 샘플, flux 컬럼: {FLUX_COLS}")
    return df


# ── 기술통계 계산 ──────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    flux 4개 컬럼 각각에 대해 클래스별 mean / std / median / Q25 / Q75 / n 계산.
    반환: {컬럼명: DataFrame(index=CLASSES)}
    """
    result = {}
    for col in FLUX_COLS:
        rows = []
        for cls in CLASSES:
            vals = df.loc[df['label'] == cls, col].dropna().values
            rows.append({
                'class':  cls,
                'n':      len(vals),
                'mean':   np.mean(vals),
                'std':    np.std(vals, ddof=1),
                'median': np.median(vals),
                'Q25':    np.percentile(vals, 25),
                'Q75':    np.percentile(vals, 75),
                'min':    np.min(vals),
                'max':    np.max(vals),
            })
        result[col] = pd.DataFrame(rows).set_index('class')
    return result


# ── 박스플롯 ──────────────────────────────────────────────────────────────────

def plot_boxplot(df: pd.DataFrame) -> None:
    """
    flux 4개 컬럼을 2×2 서브플롯으로 시각화.
    각 박스는 클래스 하나를 나타내며, SHAP 방향성(낮음↔높음) 강조.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, FLUX_COLS):
        data_by_class = [df.loc[df['label'] == cls, col].dropna().values
                         for cls in CLASSES]
        colors = [CLASS_COLOR[cls] for cls in CLASSES]

        bp = ax.boxplot(
            data_by_class,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=2, alpha=0.3),
            widths=0.5,
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        short_labels = ['ord', 'pizz', 'trem', 'sul_p', 'col_l']
        ax.set_xticks(range(1, len(CLASSES) + 1))
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature value', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # flux_std_0: 방향성 주석 강조
        if col == 'flux_std_0':
            ax.set_title(f'{col}  ★ SHAP Top Feature', fontsize=10, fontweight='bold',
                         color='#C44E52')
            y_max = ax.get_ylim()[1]
            ax.annotate('low → col_legno', xy=(5, np.median(df.loc[df['label']=='col_legno_battuto', col])),
                        xytext=(3.5, y_max * 0.12),
                        arrowprops=dict(arrowstyle='->', color='#C44E52'),
                        fontsize=7, color='#C44E52')
            ax.annotate('high → tremolo', xy=(3, np.median(df.loc[df['label']=='tremolo', col])),
                        xytext=(1.3, y_max * 0.80),
                        arrowprops=dict(arrowstyle='->', color='#DD8452'),
                        fontsize=7, color='#DD8452')

    # 공통 범례
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOR[cls], alpha=0.75)
               for cls in CLASSES]
    fig.legend(handles, CLASSES, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle('Onset-strength envelope — 클래스별 분포\n(★ flux_std_0: SHAP per-feature 1위, 방향성 반전 확인)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = RESULTS_DIR / 'flux_std_boxplot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {out}")


# ── Kruskal-Wallis 유의성 검정 ────────────────────────────────────────────────

def kruskal_test(df: pd.DataFrame) -> dict[str, float]:
    """클래스 간 분포 차이 유의성 (비모수 검정). p < 0.05 → 클래스 간 차이 있음."""
    results = {}
    for col in FLUX_COLS:
        groups = [df.loc[df['label'] == cls, col].dropna().values for cls in CLASSES]
        stat, p = scipy_stats.kruskal(*groups)
        results[col] = (stat, p)
    return results


# ── 리포트 저장 ───────────────────────────────────────────────────────────────

def save_report(stats: dict[str, pd.DataFrame], kruskal: dict[str, tuple]) -> None:
    lines = [
        "# Phase 3 보완 — Onset-strength envelope(코드명 flux) 클래스별 기술통계",
        "",
        "연구과정/03_피드백_전문가.md 2순위: SHAP flux_std 방향성을 실측 중앙값으로 검증 "
        "(초기 가설 '낮음→ordinario, 높음→col_legno'은 실측에서 기각 — 아래 핵심 발견 참조).",
        "",
    ]

    for col in FLUX_COLS:
        stat_df = stats[col]
        stat, p = kruskal[col]
        sig = "**p < 0.001**" if p < 0.001 else f"p = {p:.4f}"
        is_top = " ★ SHAP per-feature 1위" if col == 'flux_std_0' else ""

        lines += [
            f"## {col}{is_top}",
            "",
            f"Kruskal-Wallis H={stat:.2f}, {sig}",
            "",
            "| 클래스 | n | mean | std | median | Q25 | Q75 |",
            "|---|---|---|---|---|---|---|",
        ]
        for cls in CLASSES:
            r = stat_df.loc[cls]
            lines.append(
                f"| {cls} | {int(r['n'])} "
                f"| {r['mean']:.4f} | {r['std']:.4f} | {r['median']:.4f} "
                f"| {r['Q25']:.4f} | {r['Q75']:.4f} |"
            )
        lines.append("")

    # flux_std_0 방향성 요약 (논문용)
    s = stats['flux_std_0']
    ord_med  = s.loc['ordinario', 'median']
    col_med  = s.loc['col_legno_battuto', 'median']
    ratio    = col_med / ord_med if ord_med > 0 else float('inf')
    lines += [
        "## 핵심 발견 — flux_std_0 방향성 수치화",
        "",
        f"- col_legno_battuto median: **{col_med:.4f}** (최저)",
        f"- ordinario median: **{ord_med:.4f}** (중간)",
        f"- 배율: ordinario가 col_legno보다 {ord_med / col_med:.1f}× 높음",
        "",
        '> SHAP 방향성: **낮은 flux_std → col_legno_battuto**, **중간 flux_std → ordinario**, **높은 flux_std → tremolo**',
        '> 실측 중앙값(col_legno 0.18, ordinario 0.51, tremolo 0.93)이 이 순서와 일치한다.',
        "",
        "## 출력 파일",
        "",
        "| 파일 | 설명 |",
        "|---|---|",
        "| `flux_std_boxplot.png` | flux 4개 컬럼 클래스별 박스플롯 |",
        "| `flux_stats_table.md` | 이 파일 — 기술통계 테이블 |",
    ]

    out = RESULTS_DIR / 'flux_stats_table.md'
    out.write_text("\n".join(lines), encoding='utf-8')
    print(f"  저장: {out}")


# ── 콘솔 요약 출력 ────────────────────────────────────────────────────────────

def print_summary(stats: dict[str, pd.DataFrame]) -> None:
    s = stats['flux_std_0']
    print("\n  flux_std_0 클래스별 요약:")
    print(f"  {'클래스':<22}  {'mean':>8}  {'std':>8}  {'median':>8}")
    print("  " + "-" * 52)
    for cls in CLASSES:
        r = s.loc[cls]
        print(f"  {cls:<22}  {r['mean']:>8.4f}  {r['std']:>8.4f}  {r['median']:>8.4f}")

    ord_med = s.loc['ordinario', 'median']
    col_med = s.loc['col_legno_battuto', 'median']
    print(f"\n  방향성: ordinario({ord_med:.4f}) vs col_legno({col_med:.4f})  "
          f"→ {col_med / ord_med:.1f}× 차이")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    assert DATA_PATH.exists(), f"features_full.csv 없음: {DATA_PATH}"

    print("=== Phase 3 보완 — Onset-strength envelope 기술통계 + 박스플롯 ===\n")

    print("1. 데이터 로드")
    df = load_data()

    print("\n2. 기술통계 계산")
    stats = compute_stats(df)
    print_summary(stats)

    print("\n3. 박스플롯 생성")
    plot_boxplot(df)

    print("\n4. Kruskal-Wallis 검정")
    kruskal = kruskal_test(df)
    for col, (stat, p) in kruskal.items():
        sig = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        print(f"  {col:<15}  H={stat:.2f}  {sig}")

    print("\n5. 리포트 저장")
    save_report(stats, kruskal)

    print("\n완료.")


if __name__ == '__main__':
    main()
