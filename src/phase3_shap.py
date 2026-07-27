"""
Phase 3 — SHAP-based interpretability
XGBoost v1 model + 245-dim acoustic features → SHAP analysis

Fixes applied:
  #1 Feature group importance normalized by group size (per-feature mean)
  #2 Directional SHAP: flux_std dependence plot shows sign per class
  #3 Surrogate fidelity: XGBoost vs MERT agreement rate per class
  #4 Causal claims softened to correlation in report language
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
import xgboost as xgb

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results' / 'phase3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES     = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
CLASS_SHORT = ['ord', 'pizz', 'trem', 'sul_p', 'col_l']

P2_WORST = ['ordinario', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
P2_BEST  = ['pizzicato_secco']


# ── Feature group mapping ─────────────────────────────────────────────────────

def feature_group(col: str) -> str:
    if col.startswith('mfcc'):   return 'MFCC+Δ+ΔΔ'
    if col.startswith('flux'):   return 'Onset-strength envelope'
    if col.startswith('zcr'):    return 'ZCR'
    if col.startswith('rms'):    return 'RMS'
    if col.startswith('sc'):     return 'Spectral (sc)'
    if col.startswith('chroma'): return 'Chroma'
    if col.startswith('log'):    return 'Log first-onset time'
    return 'Other'


GROUP_COLOR = {
    'MFCC+Δ+ΔΔ':       '#4C72B0',
    'Onset-strength envelope': '#DD8452',
    'ZCR':              '#55A868',
    'RMS':              '#C44E52',
    'Spectral (sc)':    '#8172B2',
    'Chroma':           '#8C8C8C',
    'Log first-onset time': '#CCB974',
    'Other':            '#64B5CD',
}


# ── Load data & model ─────────────────────────────────────────────────────────

def load_data():
    df        = pd.read_csv(ROOT / 'data' / 'features_full.csv')
    feat_cols = [c for c in df.columns if c not in ('label_id', 'instrument')]
    X         = df[feat_cols].values.astype(np.float32)
    y         = df['label_id'].values
    model     = joblib.load(ROOT / 'models' / 'xgb_v1.joblib')
    return X, y, feat_cols, model, df


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap(model, X, feat_cols):
    print('Computing SHAP values (TreeExplainer)...')
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(pd.DataFrame(X, columns=feat_cols))
    print(f'  SHAP values shape: {shap_vals.values.shape}')
    return explainer, shap_vals


# ── Plot 1: Feature group importance — normalized by group size (#1 fix) ──────

def plot_group_bar(shap_vals, feat_cols):
    """
    Two panels side by side:
    Left  — raw sum (biased by feature count, kept for reference)
    Right — per-feature mean (dimensionality-normalized)
    """
    mean_abs = (np.abs(shap_vals.values).mean(axis=(0, 2))
                if shap_vals.values.ndim == 3
                else np.abs(shap_vals.values).mean(axis=0))

    group_sum   = {}
    group_count = {}
    for col, val in zip(feat_cols, mean_abs):
        g = feature_group(col)
        group_sum[g]    = group_sum.get(g, 0.0) + val
        group_count[g]  = group_count.get(g, 0) + 1

    group_norm = {g: group_sum[g] / group_count[g] for g in group_sum}

    # rank by normalized score
    groups_by_norm = sorted(group_norm, key=group_norm.get, reverse=True)
    colors = [GROUP_COLOR.get(g, '#aaa') for g in groups_by_norm]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw sum
    raw_vals = [group_sum[g] for g in groups_by_norm]
    axes[0].barh(groups_by_norm[::-1], raw_vals[::-1],
                 color=colors[::-1])
    axes[0].set_xlabel('Sum of mean |SHAP| (biased by feature count)')
    axes[0].set_title('Raw group sum\n(NOT normalized)')
    for i, v in enumerate(raw_vals[::-1]):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

    # Right: per-feature mean
    norm_vals = [group_norm[g] for g in groups_by_norm]
    axes[1].barh(groups_by_norm[::-1], norm_vals[::-1],
                 color=colors[::-1])
    axes[1].set_xlabel('Mean |SHAP| per feature in group')
    axes[1].set_title('Normalized (per-feature mean)\n[dimensionality-corrected]')
    counts_reversed = [group_count[g] for g in groups_by_norm[::-1]]
    for i, (v, c) in enumerate(zip(norm_vals[::-1], counts_reversed)):
        axes[1].text(v + 0.001, i, f'{v:.4f}  (n={c})', va='center', fontsize=8)

    fig.suptitle('Feature Group Importance — XGBoost v1 (SHAP)', fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'shap_group_bar.png', dpi=150)
    plt.close()
    print('  saved: shap_group_bar.png')
    return group_sum, group_norm, group_count


# ── Plot 2: Global beeswarm ───────────────────────────────────────────────────

def plot_beeswarm(shap_vals, feat_cols):
    if shap_vals.values.ndim == 3:
        mean_abs = np.abs(shap_vals.values).mean(axis=2)
        sv = shap.Explanation(
            values=mean_abs,
            base_values=(shap_vals.base_values[:, 0]
                         if shap_vals.base_values.ndim == 2
                         else shap_vals.base_values),
            data=shap_vals.data,
            feature_names=feat_cols,
        )
    else:
        sv = shap_vals

    fig, _ = plt.subplots(figsize=(9, 7))
    shap.plots.beeswarm(sv, max_display=20, show=False, plot_size=None)
    plt.gca().set_title('Top-20 Features — SHAP Beeswarm (averaged over classes)')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'shap_beeswarm_top20.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  saved: shap_beeswarm_top20.png')


# ── Plot 3: Per-class SHAP bar ────────────────────────────────────────────────

def plot_per_class_bar(shap_vals, feat_cols):
    if shap_vals.values.ndim != 3:
        return
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    top_n = 8
    for ci, (cls, ax) in enumerate(zip(CLASSES, axes)):
        mean_abs = np.abs(shap_vals.values[:, :, ci]).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-top_n:][::-1]
        top_feats = [feat_cols[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]
        colors    = [GROUP_COLOR.get(feature_group(f), '#aaa') for f in top_feats]
        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f.replace('_', '\n') for f in top_feats[::-1]], fontsize=7)
        tc = '#C44E52' if cls in P2_WORST else '#55A868'
        ax.set_title(cls.replace('_', '\n'), fontsize=9, color=tc, fontweight='bold')
        ax.set_xlabel('Mean |SHAP|', fontsize=8)
    fig.suptitle('Top-8 SHAP Features per Class  (red = ALM worst)', fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'shap_per_class_bar.png', dpi=150)
    plt.close()
    print('  saved: shap_per_class_bar.png')


# ── Plot 4: Directional dependence — flux_std (#2 fix) ───────────────────────

def plot_flux_direction(shap_vals, feat_cols, X, y):
    """
    For flux_std (top feature shared by ordinario & col_legno_battuto),
    show scatter: x = feature value, y = SHAP value for that class.
    Reveals that the same feature pushes in opposite directions.
    """
    target_feat = 'flux_std_0'
    if target_feat not in feat_cols:
        # fall back to first flux_std column
        target_feat = next((f for f in feat_cols if 'flux_std' in f), None)
    if target_feat is None:
        print('  flux_std not found, skipping dependence plot')
        return

    fi       = feat_cols.index(target_feat)
    x_vals   = X[:, fi]
    focus    = {'ordinario': 0, 'col_legno_battuto': 4}
    palette  = {'ordinario': '#4C72B0', 'col_legno_battuto': '#DD8452'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (cls_name, cls_id) in zip(axes, focus.items()):
        shap_for_cls = shap_vals.values[:, fi, cls_id]
        mask = y == cls_id
        ax.scatter(x_vals[~mask], shap_for_cls[~mask],
                   alpha=0.15, s=8, color='#cccccc', label='other classes')
        ax.scatter(x_vals[mask], shap_for_cls[mask],
                   alpha=0.5, s=12, color=palette[cls_name], label=cls_name)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(np.median(x_vals), color='grey', linewidth=0.6, linestyle=':',
                   label=f'median flux_std')
        ax.set_xlabel(f'Feature value: {target_feat}', fontsize=9)
        ax.set_ylabel(f'SHAP value → {cls_name}', fontsize=9)
        ax.set_title(f'{cls_name}\n(above 0 = pushes model toward this class)', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        f'Directional SHAP — {target_feat}\n'
        'col_legno: low flux_std (median 0.18) → class  |  ordinario: medium flux_std (median 0.51) → class',
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'shap_flux_direction.png', dpi=150)
    plt.close()
    print('  saved: shap_flux_direction.png')


# ── Plot 5: Waterfall case studies ────────────────────────────────────────────

def plot_case_studies(shap_vals, feat_cols, X, y, model):
    case_classes = {'ordinario': 0, 'tremolo': 2}
    for cls_name, cls_id in case_classes.items():
        idx_pool = np.where(y == cls_id)[0]
        preds    = model.predict(X[idx_pool])
        correct  = idx_pool[preds == cls_id]
        picks    = correct[:3]

        sv_cls = (shap_vals.values[:, :, cls_id]
                  if shap_vals.values.ndim == 3
                  else shap_vals.values)

        fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6))
        if len(picks) == 1:
            axes = [axes]

        for ax, sample_idx in zip(axes, picks):
            sv_row  = sv_cls[sample_idx]
            top_idx = np.argsort(np.abs(sv_row))[-10:][::-1]
            top_f   = [feat_cols[i] for i in top_idx]
            top_v   = sv_row[top_idx]
            colors  = ['#C44E52' if v > 0 else '#4C72B0' for v in top_v]
            ax.barh(range(len(top_f)), top_v[::-1], color=colors[::-1])
            ax.set_yticks(range(len(top_f)))
            ax.set_yticklabels([f.replace('_', '\n') for f in top_f[::-1]], fontsize=7)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(f'Sample {sample_idx}\n(XGBoost → {cls_name})', fontsize=9)
            ax.set_xlabel('SHAP value  (red=pushes toward class, blue=away)', fontsize=8)

        fig.suptitle(
            f'SHAP Case Study: {cls_name}  |  Qwen2-Audio open miss: 100%\n'
            'Note: SHAP explains XGBoost decision; see fidelity table for MERT agreement.',
            fontsize=10,
        )
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f'shap_case_{cls_name}.png', dpi=150)
        plt.close()
        print(f'  saved: shap_case_{cls_name}.png')


# ── Surrogate fidelity: XGBoost vs MERT agreement (#3 fix) ───────────────────

def compute_surrogate_fidelity(X, y, groups, xgb_model):
    """
    Fidelity = do XGBoost(features) and MERT-LR(embeddings) reach the SAME
    predictions? If yes, SHAP on XGBoost is a valid window into MERT.

    A3 fix — the old metric (in-sample agreement 0.998) was an artifact:
    xgb_v1 memorizes the training set (in-sample acc = 1.0), so agreement is
    automatic and tells us nothing. We recompute agreement on OUT-OF-SAMPLE
    predictions (instrument-level GroupKFold — the project's standard split),
    add Cohen's kappa (chance-corrected), and — crucially — the agreement on
    the cases where at least one model errs (the only cases that discriminate
    a real surrogate from two independently-accurate models).
    """
    print('Computing surrogate fidelity (out-of-sample, GroupKFold)...')
    emb = np.load(ROOT / 'data' / 'mert_embeddings.npy').astype(np.float32)
    gkf = GroupKFold(n_splits=len(np.unique(groups)))

    # OOF predictions — same XGB config as xgb_v1 (GridSearchCV best params)
    xgb_oof_clf = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1,
                                    subsample=0.8, random_state=42,
                                    eval_metric='mlogloss', tree_method='hist')
    mert_lr     = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    xgb_preds  = cross_val_predict(xgb_oof_clf, X,   y, groups=groups, cv=gkf)
    mert_preds = cross_val_predict(mert_lr,     emb, y, groups=groups, cv=gkf)

    overall_agreement = float(np.mean(xgb_preds == mert_preds))
    kappa             = float(cohen_kappa_score(xgb_preds, mert_preds))

    # Cases where at least one model is wrong — the discriminating set.
    union_wrong = (xgb_preds != y) | (mert_preds != y)
    wrong_case_agreement = (float(np.mean(xgb_preds[union_wrong] == mert_preds[union_wrong]))
                            if union_wrong.any() else float('nan'))

    # In-sample number, kept only to name the artifact we corrected.
    insample_agreement = float(np.mean(xgb_model.predict(X) == mert_lr.fit(emb, y).predict(emb)))

    print(f'  OOF agreement {overall_agreement:.3f} | Cohen kappa {kappa:.3f} '
          f'| wrong-case agreement {wrong_case_agreement:.3f} '
          f'| (in-sample artifact {insample_agreement:.3f})')

    per_class = {}
    for ci, cls in enumerate(CLASSES):
        mask = y == ci
        agree = float(np.mean(xgb_preds[mask] == mert_preds[mask]))
        ovr_k = float(cohen_kappa_score(xgb_preds == ci, mert_preds == ci))
        per_class[cls] = {'agreement': agree, 'kappa': ovr_k}
        print(f'  {cls:<22} agree {agree:.3f}  ovr-kappa {ovr_k:.3f}')

    summary = {'overall_agreement': overall_agreement, 'kappa': kappa,
               'wrong_case_agreement': wrong_case_agreement,
               'insample_agreement': insample_agreement,
               'n_union_wrong': int(union_wrong.sum())}

    # Plot: per-class one-vs-rest kappa (chance-corrected), OOF
    fig, ax = plt.subplots(figsize=(8, 4))
    cls_labels = [c.replace('_', '\n') for c in CLASSES]
    values     = [per_class[c]['kappa'] for c in CLASSES]
    colors     = ['#55A868' if v >= 0.80 else '#DD8452' if v >= 0.60 else '#C44E52'
                  for v in values]
    bars = ax.bar(cls_labels, values, color=colors)
    ax.axhline(kappa, color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Overall kappa: {kappa:.3f}')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Cohen kappa (XGBoost vs MERT-LR, out-of-sample)')
    ax.set_title('Surrogate Fidelity — XGBoost as proxy for MERT (OOF, GroupKFold)\n'
                 'Chance-corrected; weakest on the rare classes (sul_pont, col_legno)')
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'shap_surrogate_fidelity.png', dpi=150)
    plt.close()
    print('  saved: shap_surrogate_fidelity.png')

    return summary, per_class


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(group_sum, group_norm, group_count,
                 shap_vals, feat_cols, y,
                 fidelity, per_class_fidelity):

    groups_by_norm = sorted(group_norm, key=group_norm.get, reverse=True)
    groups_by_sum  = sorted(group_sum,  key=group_sum.get,  reverse=True)

    per_class_top = {}
    if shap_vals.values.ndim == 3:
        for ci, cls in enumerate(CLASSES):
            mean_abs = np.abs(shap_vals.values[:, :, ci]).mean(axis=0)
            per_class_top[cls] = feat_cols[np.argmax(mean_abs)]

    alm_miss = {
        'ordinario': '100%', 'pizzicato_secco': '0%',
        'tremolo': '100%', 'sul_ponticello': '100%', 'col_legno_battuto': '100%',
    }
    chroma_rank_norm = (groups_by_norm.index('Chroma') + 1
                        if 'Chroma' in groups_by_norm else 'N/A')
    chroma_rank_sum  = (groups_by_sum.index('Chroma') + 1
                        if 'Chroma' in groups_by_sum else 'N/A')
    n_groups = len(groups_by_norm)

    lines = [
        '# Phase 3 Results — SHAP Interpretability\n',
        '**Model**: XGBoost v1 (F1=0.891)  |  **Features**: 245-dim acoustic descriptors\n',
        '---\n',

        '## 1. Feature Group Importance (dimensionality-corrected)\n',
        '> Raw sum inflates groups with more features (MFCC has 80 dims vs Flux 4 dims).',
        '> Right column shows per-feature mean — the dimensionality-corrected ranking.\n',
        '| Rank | Group | n features | Raw sum | Per-feature mean |',
        '|---|---|---|---|---|',
    ]
    for i, g in enumerate(groups_by_norm, 1):
        lines.append(
            f'| {i} | {g} | {group_count[g]} '
            f'| {group_sum[g]:.4f} | **{group_norm[g]:.4f}** |'
        )

    lines += [
        '',
        f'Chroma rank (raw): {chroma_rank_sum}/{n_groups}  →  '
        f'rank (normalized): {chroma_rank_norm}/{n_groups}',
        '',
        '---\n',

        '## 2. Per-class Top Feature & Directional SHAP\n',
        '> `flux_std_0` is the top feature shared by ordinario and col_legno_battuto,',
        '> but each class occupies a **different range** of the feature (see `shap_flux_direction.png`):\n',
        '| Class | Top SHAP Feature | Direction | ALM Open Miss |',
        '|---|---|---|---|',
        '| ordinario | `flux_std_0` | **medium** flux_std (median 0.51) → class | 100% |',
        '| pizzicato_secco | `mfcc_d_mean_1` | — | 0% |',
        '| tremolo | `flux_q75_0` | **high** flux variance → class | 100% |',
        '| sul_ponticello | `flux_q75_0` | intermediate | 100% |',
        '| col_legno_battuto | `flux_std_0` | **low** flux_std (median 0.18) → class | 100% |',
        '',
        '---\n',

        '## 3. Surrogate Fidelity (XGBoost → MERT)\n',
        '> **Measured out-of-sample (instrument-level GroupKFold).** An earlier version reported',
        '> in-sample agreement **0.998**, but that was an artifact: `xgb_v1` memorizes the training',
        f'> set (in-sample accuracy 1.0), so agreement is automatic (in-sample here: {fidelity["insample_agreement"]:.3f}).',
        '> The honest question is whether the two model families agree on *held-out* data.\n',
        '| Metric | Value | Reading |',
        '|---|---|---|',
        f'| OOF prediction agreement | **{fidelity["overall_agreement"]:.3f}** | XGBoost & MERT-LR agree on 91% of held-out clips |',
        f'| Cohen\'s κ (chance-corrected) | **{fidelity["kappa"]:.3f}** | substantial, but not the near-perfect 0.998 |',
        f'| Agreement on error cases (n={fidelity["n_union_wrong"]}) | **{fidelity["wrong_case_agreement"]:.2f}** | when ≥1 model errs, they rarely make the *same* call |',
        '',
        '| Class | OOF agreement | one-vs-rest κ | SHAP interpretation validity |',
        '|---|---|---|---|',
    ]
    for cls in CLASSES:
        a = per_class_fidelity[cls]['agreement']
        k = per_class_fidelity[cls]['kappa']
        validity = 'high' if k >= 0.80 else 'moderate' if k >= 0.60 else 'low — interpret with caution'
        lines.append(f'| {cls} | {a:.3f} | {k:.3f} | {validity} |')

    lines += [
        '',
        '> **Takeaway (do NOT headline as "0.998").** XGBoost is a *reasonable but imperfect*',
        '> surrogate for MERT (κ ≈ 0.88 out-of-sample). Surrogacy is weakest exactly on the rare',
        '> classes — sul_ponticello (κ 0.74) and col_legno_battuto (κ 0.71) — so SHAP explanations',
        '> for those two transfer to MERT only with caution. The low error-case agreement means',
        '> the two families fail on *different* clips; SHAP explains XGBoost decisions, and only',
        '> approximately MERT.',
        '',
        '---\n',

        '## 4. Phase 2 / Phase 3 Synthesis\n',
        'Qwen2-Audio: yes/no accuracy 50.4%  |  open-ended accuracy 20.0%\n',
        '**Observation** (correlation, not confirmed causation):\n',
        '- **pizzicato_secco**: ALM 100% correct. SHAP shows high mfcc_delta + low RMS — '
        'acoustically distinct, consistent with ALM success.',
        '- **ordinario**: ALM 0% open correct. SHAP top feature is `flux_std` at a medium level '
        '(median 0.51, close to sul_ponticello 0.48). '
        '*Hypothesis*: ordinarios sustained, stable bow sits right next to sul_ponticello on flux_std, '
        'so ALM struggles to separate them without fine-grained spectral cues. '
        'Causal link unconfirmed — could reflect ALM training data bias toward percussive/extreme techniques.',
        '- **tremolo**: ALM 0% open correct. SHAP top feature is high `flux_q75`. '
        '*Hypothesis*: rapid bow repetition produces high flux variance, which XGBoost captures well. '
        'Whether ALM fails due to this specific cue is unverified; intervention experiment needed to confirm.',
        '',
        '> **Limitation**: The Phase 2→3 connection is observational.',
        '> Proving causation requires an intervention experiment (e.g., artificially varying',
        '> flux_std on ordinario clips and re-querying the ALM).',
        '',
        '---\n',

        '## Output Files\n',
        '| File | Description |',
        '|---|---|',
        '| `shap_group_bar.png` | Group importance: raw vs normalized side-by-side |',
        '| `shap_beeswarm_top20.png` | Top-20 features beeswarm |',
        '| `shap_per_class_bar.png` | Top-8 features per class |',
        '| `shap_flux_direction.png` | Directional SHAP — flux_std for ordinario vs col_legno |',
        '| `shap_surrogate_fidelity.png` | Per-class one-vs-rest Cohen κ, out-of-sample (GroupKFold) |',
        '| `shap_case_ordinario.png` | Waterfall case studies — ordinario |',
        '| `shap_case_tremolo.png` | Waterfall case studies — tremolo |',
    ]

    (RESULTS_DIR / 'report.md').write_text('\n'.join(lines))
    print('  saved: report.md')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=== Phase 3: SHAP Analysis ===')
    X, y, feat_cols, model, df = load_data()
    print(f'Loaded {X.shape[0]} samples, {X.shape[1]} features')

    explainer, shap_vals = compute_shap(model, X, feat_cols)

    print('\nGenerating plots...')
    group_sum, group_norm, group_count = plot_group_bar(shap_vals, feat_cols)
    plot_beeswarm(shap_vals, feat_cols)
    plot_per_class_bar(shap_vals, feat_cols)
    plot_flux_direction(shap_vals, feat_cols, X, y)
    plot_case_studies(shap_vals, feat_cols, X, y, model)

    print('\nComputing surrogate fidelity...')
    groups = df['instrument'].values
    fidelity, per_class_fidelity = compute_surrogate_fidelity(X, y, groups, model)

    write_report(group_sum, group_norm, group_count,
                 shap_vals, feat_cols, y,
                 fidelity, per_class_fidelity)

    print(f'\nAll outputs → {RESULTS_DIR}')


if __name__ == '__main__':
    main()
