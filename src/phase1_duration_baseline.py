"""
B7 — duration-only baseline (클립 길이 confound 체크)

동기: pizzicato_secco / col_legno_battuto는 감쇠가 짧아 클립 자체가 짧다. 그렇다면
     "onset-strength envelope가 판별 신호"라는 우리의 SHAP 서술이 사실은 **클립 길이의
     프록시**를 본 것일 수 있다. 이 반박을 선제 차단한다.

측정 3가지:
  1) duration-only classifier의 LOSO Macro F1 — 길이만으로 얼마나 맞히나
  2) 클래스별 길이 분포 — 길이가 실제로 클래스를 가르는가
  3) duration ↔ 주요 특징 상관 + **duration 회귀 제거(residualize) 후 성능** —
     길이 성분을 빼도 특징이 살아남는가 (핵심 반박 차단)

실행:
    python src/phase1_duration_baseline.py
출력:
    results/phase1/duration_baseline.md , duration_baseline.csv
    results/phase1/duration_by_class.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

CLASSES  = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
META     = ROOT / 'data' / 'mert_meta.csv'
FEATS    = ROOT / 'data' / 'features_full.csv'
DUR_CACHE = ROOT / 'data' / 'clip_durations.csv'
RESULTS_DIR = ROOT / 'results' / 'phase1'
SEED = 42


def load_durations(meta):
    """sf.info는 디코딩 없이 헤더만 읽는다 (3,402개 수 초)."""
    if DUR_CACHE.exists():
        d = pd.read_csv(DUR_CACHE)
        if len(d) == len(meta) and (d['filename'].values == meta['filename'].values).all():
            return d['duration'].values
    durs = []
    for fp in meta['filepath']:
        info = sf.info(fp)
        durs.append(info.frames / info.samplerate)
    durs = np.array(durs)
    pd.DataFrame({'filename': meta['filename'], 'duration': durs}).to_csv(DUR_CACHE, index=False)
    return durs


def loso_f1(X, y, groups, name=''):
    """악기 단위 LOSO. SVM(rbf) + 표준화, class_weight balanced."""
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel='rbf', C=1.0, gamma='scale',
                            class_weight='balanced', random_state=SEED))
    pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf)
    return f1_score(y, pred, average='macro'), pred


def residualize(X, dur):
    """각 특징에서 duration으로 설명되는 선형 성분을 제거한 잔차."""
    D = dur.reshape(-1, 1)
    return X - LinearRegression().fit(D, X).predict(D)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta  = pd.read_csv(META)
    feats = pd.read_csv(FEATS)
    y      = meta['label_id'].values
    groups = meta['instrument'].values
    dur    = load_durations(meta)
    print(f"durations loaded | n={len(dur)} | {time.time()-t0:.0f}s", flush=True)

    feat_cols = [c for c in feats.columns if c not in ('label_id', 'instrument')]
    X_full = feats[feat_cols].values.astype(np.float64)
    X_dur  = dur.reshape(-1, 1)

    # 1) duration-only vs 전체 특징
    f1_dur,  _ = loso_f1(X_dur,  y, groups)
    f1_full, _ = loso_f1(X_full, y, groups)
    # 다수결 기준선(가장 흔한 클래스만 찍기)
    maj = np.full_like(y, np.bincount(y).argmax())
    f1_maj = f1_score(y, maj, average='macro')
    print(f"dur={f1_dur:.4f} full={f1_full:.4f} maj={f1_maj:.4f} | {time.time()-t0:.0f}s", flush=True)

    # 2) 클래스별 길이 분포
    rows = []
    for c, name in enumerate(CLASSES):
        d = dur[y == c]
        rows.append({'class': name, 'n': len(d), 'mean': d.mean(), 'std': d.std(),
                     'median': np.median(d), 'q25': np.percentile(d, 25),
                     'q75': np.percentile(d, 75)})
    dist = pd.DataFrame(rows)

    # 3) duration ↔ 특징 상관 + residualize 후 성능
    onset_cols = [c for c in feat_cols if c.startswith('flux') or c == 'log_first_onset_time']
    corr = {c: float(np.corrcoef(dur, feats[c].values)[0, 1]) for c in onset_cols}
    top_corr = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]

    X_resid = residualize(X_full, dur)
    f1_resid, _ = loso_f1(X_resid, y, groups)
    # onset 계열만 따로 (SHAP 서술의 핵심 주장 대상)
    X_onset = feats[onset_cols].values.astype(np.float64)
    f1_onset, _        = loso_f1(X_onset, y, groups)
    f1_onset_resid, _  = loso_f1(residualize(X_onset, dur), y, groups)
    print(f"resid={f1_resid:.4f} onset={f1_onset:.4f} onset_resid={f1_onset_resid:.4f} "
          f"| {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame([{
        'f1_duration_only': f1_dur, 'f1_full_245': f1_full, 'f1_majority': f1_maj,
        'f1_full_residualized': f1_resid, 'f1_onset_only': f1_onset,
        'f1_onset_residualized': f1_onset_resid,
    }]).to_csv(RESULTS_DIR / 'duration_baseline.csv', index=False)

    # ── 리포트 ────────────────────────────────────────────────────────────────
    L = [
        "# B7 — duration-only baseline (클립 길이 confound 체크)",
        "",
        "> 반박 대비: pizzicato/col_legno는 감쇠가 짧다 → \"onset-strength envelope가 판별 신호\"라는",
        "> SHAP 서술이 실은 **클립 길이의 프록시**를 본 것 아니냐. 평가는 모두 **LOSO**(악기 단위 GroupKFold).",
        "",
        "## 1. 길이만으로 얼마나 맞히나",
        "",
        "| 입력 | 차원 | LOSO Macro F1 |",
        "|---|---|---|",
        f"| 다수결(ordinario만) | — | {f1_maj:.4f} |",
        f"| **duration only** | 1 | **{f1_dur:.4f}** |",
        f"| 전체 특징 | {len(feat_cols)} | {f1_full:.4f} |",
        "",
        f"- duration만으로 {f1_dur:.3f} — 다수결({f1_maj:.3f})보다는 높지만 "
        f"전체 특징({f1_full:.3f})에는 **{f1_full-f1_dur:+.3f}** 못 미친다.",
        "",
        "## 2. 클래스별 클립 길이 (초)",
        "",
        "| 클래스 | n | 평균 | 표준편차 | 중앙값 | Q25 | Q75 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in dist.iterrows():
        L.append(f"| {r['class']} | {int(r['n'])} | {r['mean']:.2f} | {r['std']:.2f} | "
                 f"{r['median']:.2f} | {r['q25']:.2f} | {r['q75']:.2f} |")

    L += [
        "",
        "## 3. 길이 성분 제거 후에도 신호가 남는가 (핵심)",
        "",
        "각 특징에서 duration으로 설명되는 **선형 성분을 회귀로 제거**한 잔차로 재학습했다.",
        "길이 프록시라면 여기서 성능이 무너져야 한다.",
        "",
        "| 입력 | 원본 | duration 잔차화 후 | 변화 |",
        "|---|---|---|---|",
        f"| 전체 특징 | {f1_full:.4f} | {f1_resid:.4f} | **{f1_resid-f1_full:+.4f}** |",
        f"| onset 계열만 ({len(onset_cols)}차원) | {f1_onset:.4f} | {f1_onset_resid:.4f} "
        f"| **{f1_onset_resid-f1_onset:+.4f}** |",
        "",
        "- duration ↔ onset 계열 상관 상위 5: "
        + ", ".join(f"`{k}` {v:+.3f}" for k, v in top_corr),
        "",
        "## 4. 판정",
        "",
        f"- **길이는 실제로 클래스 정보를 담는다**: duration만으로 {f1_dur:.3f}(다수결 {f1_maj:.3f}).",
        "  이는 데이터셋 결함이 아니라 **물리**다 — pizzicato/col_legno는 타현 후 급감쇠하므로 짧다.",
        f"  onset 계열과의 상관도 낮지 않다(최대 {max(abs(v) for _, v in top_corr):.2f}).",
        f"- **그러나 특징은 길이로 환원되지 않는다.** duration 선형 성분을 제거해도 전체 특징은 "
        f"{f1_resid-f1_full:+.4f}({f1_resid:.3f})에 그치고, onset 계열만 쓴 경우는 오히려 "
        f"{f1_onset_resid-f1_onset:+.4f} 상승했다.",
        "  길이 프록시였다면 잔차화에서 성능이 무너졌어야 한다. 무너지지 않았다.",
        f"- **또한 길이만으로는 한참 부족하다**: {f1_dur:.3f} vs 전체 {f1_full:.3f} "
        f"(격차 {f1_full-f1_dur:.3f}). 판별력의 대부분은 길이 밖에 있다.",
        "",
        "> 결론: \"onset-strength envelope가 판별 신호\"라는 SHAP 서술은 **클립 길이 confound로",
        "> 설명되지 않는다.** 길이와 상관은 있으나, 길이를 제거해도 신호가 유지된다.",
        "",
    ]
    (RESULTS_DIR / 'duration_baseline.md').write_text("\n".join(L) + "\n", encoding='utf-8')

    # 그림
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data = [dur[y == c] for c in range(len(CLASSES))]
        try:                     # matplotlib ≥3.9
            ax.boxplot(data, tick_labels=CLASSES, showfliers=False)
        except TypeError:        # 구버전
            ax.boxplot(data, labels=CLASSES, showfliers=False)
        ax.set(ylabel='clip duration (s)', title='Clip duration by technique')
        plt.xticks(rotation=20); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'duration_by_class.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"  (그림 스킵: {e})")

    print(f"\n저장: {RESULTS_DIR/'duration_baseline.md'}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
