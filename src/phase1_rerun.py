"""
Phase 1 — 누수 수정본 headless 재실행 (CLAP 재추론 생략)

notebooks/04_phase1.ipynb의 누수 수정된 self-training 로직을 그대로 재현한다.
CLAP 제로샷은 변하지 않으므로 재추론하지 않고 기존 결과(results/phase1/clap_zeroshot.csv)를 읽는다.
※ 노트북의 cell-st-func / cell-full-run / cell-summary와 동일 로직을 유지할 것.

실행:
    python src/phase1_rerun.py            # 전체 (48 splits × 3 τ)
    python src/phase1_rerun.py --limit 3  # 앞 3 split만 (타이밍 측정용)

출력:
    results/phase1/all_results.csv   (누수 수정: iteration은 val_select F1로 선택)
    results/phase1/report.md         (4-fold 평균±표준편차, 단일 fold 헤드라인 제거)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

CLASSES = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
N_CLASSES = len(CLASSES)
TAU_LIST  = [0.7, 0.8, 0.9]
MAX_ITER  = 5

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 재현성 고정 (지시서 §6). 기존 leaked all_results.csv는 seed 미고정 런이었음.
torch.manual_seed(42)
np.random.seed(42)

EMB_PATH    = ROOT / 'data' / 'mert_embeddings.npy'
LAB_PATH    = ROOT / 'data' / 'mert_labels.npy'
SPLITS_PATH = ROOT / 'data' / 'split_configs.json'
RESULTS_DIR = ROOT / 'results' / 'phase1'

# 전역 (train_head에서 참조)
embeddings = np.load(EMB_PATH)
labels     = np.load(LAB_PATH)
counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
class_weights = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32).to(DEVICE)


# ── 모델 / 유틸 (노트북 cell-head-model, cell-train-utils와 동일) ────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim=768, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_tensors(idx):
    X = torch.tensor(embeddings[idx], dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels[idx],     dtype=torch.long).to(DEVICE)
    return X, y


def train_head(labeled_idx, val_idx, epochs=150, lr=1e-3, patience=15):
    model = LinearHead().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr, y_tr   = make_tensors(labeled_idx)
    X_val, y_val = make_tensors(val_idx)

    best_f1, best_state, wait = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val.cpu().numpy(), preds, average='macro', zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1, None


def evaluate(model, test_idx):
    model.eval()
    X_te, y_te = make_tensors(test_idx)
    with torch.no_grad():
        preds = model(X_te).argmax(1).cpu().numpy()
    y_true = y_te.cpu().numpy()
    macro   = f1_score(y_true, preds, average='macro', zero_division=0)
    per_cls = f1_score(y_true, preds, average=None, zero_division=0, labels=list(range(N_CLASSES)))
    cm      = confusion_matrix(y_true, preds, labels=list(range(N_CLASSES)))
    return macro, per_cls, cm


def split_val(labeled_idx, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(labeled_idx)
    lab = labels[idx]
    tr, va = [], []
    for c in range(N_CLASSES):
        pos = idx[lab == c]
        n_val = max(1, round(len(pos) * val_ratio))
        chosen = rng.choice(len(pos), n_val, replace=False)
        va.extend(pos[chosen].tolist())
        tr.extend(np.delete(pos, chosen).tolist())
    return tr, va


# ── Self-Training (누수 수정본 — 노트북 cell-st-func와 동일) ─────────────────────

def self_training(split, tau, max_iter=MAX_ITER, verbose=False):
    # iteration-선택 전용 val (원본 seed에서만, 1회, 고정) — test/unlabeled/early-stop과 분리
    seed_core, val_select = split_val(
        split['labeled_idx'], val_ratio=0.2, seed=split.get('trial', 0) + 1000
    )
    labeled_pool  = list(seed_core)
    unlabeled_idx = np.array(split['unlabeled_idx'])
    test_idx      = split['test_idx']

    iter_f1, val_sel_f1 = [], []
    iter_acc, iter_pseudo_acc = [], []

    for it in range(max_iter):
        tr_idx, es_val_idx = split_val(labeled_pool, seed=it)
        model, _, _ = train_head(tr_idx, es_val_idx)

        macro, _, cm = evaluate(model, test_idx)
        iter_f1.append(round(macro, 4))
        vsel, _, _ = evaluate(model, val_select)
        val_sel_f1.append(round(vsel, 4))

        if len(unlabeled_idx) == 0:
            iter_acc.append(0.0); iter_pseudo_acc.append(None)
            break

        model.eval()
        X_ul = torch.tensor(embeddings[unlabeled_idx], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(X_ul), dim=1).cpu().numpy()
        max_prob, pseudo_lbl = probs.max(axis=1), probs.argmax(axis=1)

        accept_mask = max_prob >= tau
        n_accept    = accept_mask.sum()
        accept_rate = n_accept / len(unlabeled_idx)
        iter_acc.append(round(float(accept_rate), 4))

        if n_accept > 0:
            oracle = labels[unlabeled_idx[accept_mask]]
            iter_pseudo_acc.append(round(float((pseudo_lbl[accept_mask] == oracle).mean()), 4))
        else:
            iter_pseudo_acc.append(None)

        if verbose:
            print(f'    iter {it}: test={macro:.4f} val_sel={vsel:.4f} 채택={n_accept} ({accept_rate:.1%})')

        labeled_pool  = labeled_pool + unlabeled_idx[accept_mask].tolist()
        unlabeled_idx = unlabeled_idx[~accept_mask]
        if accept_rate < 0.01:
            break

    sel_iter = int(np.argmax(val_sel_f1)) if val_sel_f1 else -1
    return {
        'iter_f1': iter_f1, 'val_sel_f1': val_sel_f1, 'sel_iter': sel_iter,
        'iter_acceptance': iter_acc, 'iter_pseudo_acc': iter_pseudo_acc,
        'n_iters': len(iter_f1),
    }


# ── 전체 실행 (노트북 cell-full-run과 동일) ─────────────────────────────────────

def run_full(limit=None):
    with open(SPLITS_PATH) as f:
        split_cfg = json.load(f)
    splits = split_cfg['splits']
    if limit:
        splits = splits[:limit]

    print(f"device={DEVICE} | splits={len(splits)} × τ={len(TAU_LIST)}  (embeddings={embeddings.shape})")
    all_results = []
    t0 = time.time()
    done = 0
    total = len(splits) * len(TAU_LIST)

    for s in splits:
        tr_idx, val_idx = split_val(s['labeled_idx'], seed=s['trial'])
        model, _, _ = train_head(tr_idx, val_idx)
        base_macro, base_per, _ = evaluate(model, s['test_idx'])

        for tau in TAU_LIST:
            result = self_training(s, tau=tau)
            sel = result['sel_iter']
            all_results.append({
                'fold': s['fold'], 'test_instrument': s['test_instrument'],
                'ratio': s['seed_ratio'], 'trial': s['trial'], 'tau': tau,
                'baseline_f1': round(base_macro, 4),
                'best_st_f1':  result['iter_f1'][sel] if result['iter_f1'] else round(base_macro, 4),
                'sel_iter': sel, 'n_iters': result['n_iters'],
                'final_acceptance': result['iter_acceptance'][-1] if result['iter_acceptance'] else 0,
                'iter_f1_json':    json.dumps(result['iter_f1']),
                'val_sel_f1_json': json.dumps(result['val_sel_f1']),
                **{f'base_f1_{cls}': round(v, 4) for cls, v in zip(CLASSES, base_per)},
            })
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{done:>3}/{total}] fold={s['fold']} r={s['seed_ratio']:.0%} "
                  f"t={s['trial']} τ={tau} | base={base_macro:.3f} "
                  f"st={all_results[-1]['best_st_f1']:.3f} sel_it={sel} "
                  f"| elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    df_all = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / ('all_results_smoke.csv' if limit else 'all_results.csv')
    df_all.to_csv(out_csv, index=False)
    print(f"\n저장: {out_csv}  ({len(df_all)}행, {time.time()-t0:.0f}s)")
    return df_all, bool(limit)


# ── 리포트 (노트북 cell-summary와 동일: 4-fold 평균±표준편차) ────────────────────

def write_report(df_all):
    v1_svm = 0.916
    try:
        clap_df = pd.read_csv(RESULTS_DIR / 'clap_zeroshot.csv')
        clap_macro = f1_score(clap_df['true_label'], clap_df['pred_label'],
                              average='macro', zero_division=0)
        clap_line = f'CLAP zero-shot baseline : {clap_macro:.4f}\n'
    except Exception:
        clap_line = ''

    best = df_all.groupby(['ratio', 'tau'])[['baseline_f1', 'best_st_f1']].mean().round(4)

    lines_findings = []
    for ratio in sorted(df_all['ratio'].unique()):
        best_tau_r, best_m, best_s = None, -1.0, 0.0
        for tau in sorted(df_all['tau'].unique()):
            fold_means = (df_all[(df_all['ratio'] == ratio) & (df_all['tau'] == tau)]
                          .groupby('fold')['best_st_f1'].mean())
            if fold_means.mean() > best_m:
                best_m, best_s, best_tau_r = fold_means.mean(), fold_means.std(), tau
        lines_findings.append(
            f'- seed {int(ratio*100)}%: best τ={best_tau_r}, '
            f'self-training F1={best_m:.4f}±{best_s:.4f} (fold-level, n=4)\n'
        )

    fm50 = df_all[df_all['ratio'] == 0.5].groupby('fold')['baseline_f1'].mean()
    mert_full, mert_full_std = fm50.mean(), fm50.std()

    lines = [
        '# Phase 1 Results Summary\n\n', clap_line,
        f'v1 SVM baseline         : {v1_svm}\n\n',
        '> 누수 수정: iteration은 val_select F1로 선택(test 미사용). '
        '전 수치 4-fold 평균±표준편차, 단일 fold 헤드라인 제거.\n\n',
        '## Baseline vs Self-Training (mean over folds x trials)\n\n',
        best.to_markdown(), '\n\n',
        '## Key Findings (fold-level 평균±표준편차, n=4)\n\n',
        *lines_findings,
        f'- MERT head (seed=50%): F1={mert_full:.4f}±{mert_full_std:.4f} '
        f'({"exceeds" if mert_full > v1_svm else "below"} v1 SVM {v1_svm})\n',
        '- Self-training vs baseline 향상폭의 통계적 유의성은 '
        '`phase1_significance.py` 참조 (fold n=4, 검정력 한계 있음)\n',
    ]
    (RESULTS_DIR / 'report.md').write_text(''.join(lines), encoding='utf-8')
    print(f"저장: {RESULTS_DIR / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='앞 N split만 (타이밍용)')
    args = ap.parse_args()

    df_all, is_smoke = run_full(limit=args.limit)
    if not is_smoke:
        write_report(df_all)
    else:
        print("[smoke] report 생략 (all_results_smoke.csv만 저장)")


if __name__ == '__main__':
    main()
