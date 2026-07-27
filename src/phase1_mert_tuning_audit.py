"""
작업 1 — MERT 튜닝 공정성 감사 (반례 논문 방어)

목적: "MERT는 대충 학습해서 SVM에 진 것 아니냐"는 반박을 선제 차단.
방법: MERT head에 SVM보다 큰 val 기반 하이퍼파라미터 탐색을 부여하고,
      SVM에도 동등한 val 기반 탐색(C/gamma)을 부여해 공정 재비교.
      모든 탐색은 labeled seed의 val split에서만 (test 누수 없음).

A1 수정 (2026-07-16) — val 이중사용 제거:
      이전 판은 같은 val로 early-stop 체크포인트 선택과 24-config 선택을 둘 다 수행했다.
      SVM은 early-stop이 없어 val을 선택에만 써서, MERT만 val을 이중으로 쓰는 비대칭이었다.
      method_leakage_note.md 1.4절과 phase1_rerun.py의 컨벤션대로 3분할로 교정:
        labeled → val_select 20% (config 선택 전용, 고정) + seed_core 80%
        seed_core → tr 80% + es_val 20% (early-stop 전용)
      SVM은 es_val이 불필요하므로 두 조건을 모두 보고한다:
        svm_tuned      — tr로 학습 (MERT와 학습 데이터 동일, 주 수치)
        svm_tuned_full — seed_core로 학습 (SVM 학습량을 이전 판과 동일하게 유지, 민감도)

실행:
    python src/phase1_mert_tuning_audit.py
    python src/phase1_mert_tuning_audit.py --ratios 0.5   # 특정 ratio만

출력:
    results/phase1/mert_tuning_audit.md
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

torch.manual_seed(42)
np.random.seed(42)

CLASSES   = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
N_CLASSES = len(CLASSES)
DEVICE    = 'mps' if torch.backends.mps.is_available() else 'cpu'

EMB_PATH    = ROOT / 'data' / 'mert_embeddings.npy'
LAB_PATH    = ROOT / 'data' / 'mert_labels.npy'
FEAT_PATH   = ROOT / 'data' / 'features_full.csv'
SPLITS_PATH = ROOT / 'data' / 'split_configs.json'
OUT         = ROOT / 'results' / 'phase1' / 'mert_tuning_audit.md'

# ── 데이터 (embeddings와 features_full은 행 정렬 일치 확인됨) ────────────────────
embeddings = np.load(EMB_PATH).astype(np.float32)
labels     = np.load(LAB_PATH).astype(np.int64)
_feat_df   = pd.read_csv(FEAT_PATH)
_feat_cols = [c for c in _feat_df.columns if c not in ('label_id', 'instrument')]
X_feat     = _feat_df[_feat_cols].to_numpy(dtype=np.float32)

counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
class_weights = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32).to(DEVICE)

# ── 탐색 그리드 ─────────────────────────────────────────────────────────────────
MERT_GRID = {
    'arch':    ['linear', 'mlp256'],
    'lr':      [3e-4, 1e-3, 3e-3],
    'wd':      [1e-4, 1e-3],
    'dropout': [0.1, 0.3],
}   # 2×3×2×2 = 24 config
MERT_DEFAULT = {'arch': 'linear', 'lr': 1e-3, 'wd': 1e-4, 'dropout': 0.3}  # 기존 고정값

SVM_GRID = {
    'C':     [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
}   # 4×3 = 12 config
SVM_DEFAULT = {'C': 10, 'gamma': 'scale'}  # 기존 고정값


# ── MERT head ─────────────────────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, arch, dropout, in_dim=768, n_classes=N_CLASSES):
        super().__init__()
        if arch == 'linear':
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, n_classes))
        elif arch == 'mlp256':
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, 256), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(256, n_classes))
        else:
            raise ValueError(arch)

    def forward(self, x):
        return self.net(x)


def _tensors(idx):
    return (torch.tensor(embeddings[idx]).to(DEVICE),
            torch.tensor(labels[idx]).to(DEVICE))


def train_head_cfg(tr_idx, es_idx, cfg, epochs=200, patience=20):
    """es_idx는 early-stop 체크포인트 선택에만 쓴다. config 선택은 호출자가
    별도의 val_select에서 하며, 두 집합은 겹치지 않는다 (A1)."""
    model = Head(cfg['arch'], cfg['dropout']).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=class_weights)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    X_tr, y_tr = _tensors(tr_idx)
    X_es, y_es = _tensors(es_idx)

    best_f1, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(X_tr), y_tr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_es).argmax(1).cpu().numpy()
        vf1 = f1_score(y_es.cpu().numpy(), pred, average='macro', zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, wait = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1   # best_f1 = early-stop val 점수 — config 선택에 쓰지 말 것


def eval_head(model, test_idx):
    model.eval()
    X_te, y_te = _tensors(test_idx)
    with torch.no_grad():
        pred = model(X_te).argmax(1).cpu().numpy()
    return f1_score(y_te.cpu().numpy(), pred, average='macro', zero_division=0)


# ── SVM ───────────────────────────────────────────────────────────────────────

def train_eval_svm(tr_idx, sel_idx, test_idx, cfg):
    """sel_idx = val_select. SVM은 early-stop이 없어 val을 config 선택에만 쓴다."""
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=cfg['C'], gamma=cfg['gamma'],
                                 class_weight='balanced', random_state=42))])
    pipe.fit(X_feat[tr_idx], labels[tr_idx])
    vf1 = f1_score(labels[sel_idx], pipe.predict(X_feat[sel_idx]), average='macro', zero_division=0)
    tf1 = f1_score(labels[test_idx], pipe.predict(X_feat[test_idx]), average='macro', zero_division=0)
    return vf1, tf1


def svm_default_and_tuned(tr_idx, sel_idx, test_idx):
    """기본 config test F1, 12-config 탐색 후 test F1, 선택된 config를 돌려준다."""
    _, def_test = train_eval_svm(tr_idx, sel_idx, test_idx, SVM_DEFAULT)
    best_vf1, best_test, best_cfg = -1.0, None, None
    for combo in itertools.product(*SVM_GRID.values()):
        cfg = dict(zip(SVM_GRID.keys(), combo))
        vf1, tf1 = train_eval_svm(tr_idx, sel_idx, test_idx, cfg)
        if vf1 > best_vf1:
            best_vf1, best_test, best_cfg = vf1, tf1, cfg
    return def_test, best_test, best_cfg


# ── val 분할 (stratified) ───────────────────────────────────────────────────────

def split_val(labeled_idx, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(labeled_idx); lab = labels[idx]
    tr, va = [], []
    for c in range(N_CLASSES):
        pos = idx[lab == c]
        n_val = max(1, round(len(pos) * val_ratio))
        chosen = rng.choice(len(pos), n_val, replace=False)
        va.extend(pos[chosen].tolist()); tr.extend(np.delete(pos, chosen).tolist())
    return tr, va


# ── 한 split 처리: val 기반 탐색 → test 평가 ─────────────────────────────────────

def process_split(s):
    # A1: config 선택 전용 val을 먼저 고정 (phase1_rerun.py와 동일한 seed 컨벤션),
    #     early-stop val은 남은 seed_core 안에서만 뗀다 → 두 val이 겹치지 않는다.
    seed_core, val_sel = split_val(s['labeled_idx'], val_ratio=0.2, seed=s['trial'] + 1000)
    tr_idx, es_idx     = split_val(seed_core, val_ratio=0.2, seed=s['trial'])
    test_idx = s['test_idx']
    out = {'fold': s['fold'], 'ratio': s['seed_ratio'], 'trial': s['trial'],
           'n_tr': len(tr_idx), 'n_es': len(es_idx), 'n_val_sel': len(val_sel),
           'n_seed_core': len(seed_core)}

    # MERT: 기본 config (early-stop만, 탐색 없음)
    m_def, _ = train_head_cfg(tr_idx, es_idx, MERT_DEFAULT)
    out['mert_default'] = eval_head(m_def, test_idx)

    # MERT: 24-config 탐색 — early-stop은 es_idx, config 선택은 val_sel (분리)
    best_vf1, best_test = -1.0, None
    for combo in itertools.product(*MERT_GRID.values()):
        cfg = dict(zip(MERT_GRID.keys(), combo))
        model, _ = train_head_cfg(tr_idx, es_idx, cfg)
        vf1 = eval_head(model, val_sel)
        if vf1 > best_vf1:
            best_vf1, best_test = vf1, eval_head(model, test_idx)
            out['mert_best_cfg'] = cfg
    out['mert_sel_f1'] = best_vf1
    out['mert_tuned']  = best_test

    # SVM (주 수치): MERT와 학습 데이터 동일(tr_idx)
    out['svm_default'], out['svm_tuned'], out['svm_best_cfg'] = \
        svm_default_and_tuned(tr_idx, val_sel, test_idx)

    # SVM (민감도): seed_core 전체로 학습 — es_val이 불필요한 이점을 살린 조건
    out['svm_default_full'], out['svm_tuned_full'], out['svm_best_cfg_full'] = \
        svm_default_and_tuned(seed_core, val_sel, test_idx)

    return out


# ── 집계 & 리포트 ────────────────────────────────────────────────────────────────

def fold_stats(df, ratio, col):
    fm = df[df['ratio'] == ratio].groupby('fold')[col].mean()
    return fm.mean(), fm.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='*', default=[0.05, 0.10, 0.20, 0.50])
    args = ap.parse_args()

    with open(SPLITS_PATH) as f:
        splits = json.load(f)['splits']
    splits = [s for s in splits if s['seed_ratio'] in args.ratios]

    n_mert = int(np.prod([len(v) for v in MERT_GRID.values()]))
    n_svm  = int(np.prod([len(v) for v in SVM_GRID.values()]))
    print(f"device={DEVICE} | splits={len(splits)} | MERT grid={n_mert}, SVM grid={n_svm}")

    rows, t0 = [], time.time()
    for i, s in enumerate(splits, 1):
        r = process_split(s)
        rows.append(r)
        print(f"  [{i:>2}/{len(splits)}] fold={s['fold']} r={s['seed_ratio']:.0%} t={s['trial']} | "
              f"MERT def={r['mert_default']:.3f} tuned={r['mert_tuned']:.3f} | "
              f"SVM def={r['svm_default']:.3f} tuned={r['svm_tuned']:.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)

    # 비교표
    n_tr, n_es, n_sel = df['n_tr'].min(), df['n_es'].min(), df['n_val_sel'].min()
    lines = [
        "# 작업 1 — MERT 튜닝 공정성 감사 결과",
        "",
        "> **A1 재실행본 (2026-07-16).** 이전 판은 같은 val로 early-stop과 config 선택을 둘 다",
        "> 수행해 MERT만 val을 이중사용했다 (SVM은 early-stop이 없어 비대칭). 아래는 두 val을",
        "> 분리해 재실행한 결과다. 이전 판: `mert_tuning_audit_before_A1.md`",
        "",
        "## 분할 프로토콜 (A1 교정)",
        "",
        "```",
        "labeled seed ─┬─ val_select 20%  → config 선택 전용 (고정, seed=trial+1000)",
        "              └─ seed_core  80% ─┬─ tr     80% → 학습",
        "                                 └─ es_val 20% → early-stop 전용",
        "```",
        "",
        "val_select ∩ es_val = ∅, 둘 다 test와 무관. `method_leakage_note.md` 1.4절 및",
        f"`phase1_rerun.py`의 컨벤션과 동일. 최소 규모(5% seed): tr={n_tr}, es_val={n_es}, val_select={n_sel}.",
        "",
        "SVM은 early-stop이 없어 es_val이 불필요하므로 두 조건을 모두 보고한다:",
        "",
        "| 조건 | SVM 학습 데이터 | 의미 |",
        "|---|---|---|",
        "| **SVM(튜닝)** — 주 수치 | `tr` (labeled의 64%) | MERT와 **학습 데이터 동일** — 동등 예산 비교의 기준 |",
        "| SVM(튜닝, seed_core) — 민감도 | `seed_core` (labeled의 80%) | SVM 학습량을 이전 판과 동일하게 유지 |",
        "",
        "## 튜닝 예산 (val 기반, test 누수 없음)",
        "",
        "| 모델 | 기존 | 이번 감사 탐색 |",
        "|---|---|---|",
        f"| SVM | C=10, gamma=scale (하드코딩, 탐색 0) | C∈{SVM_GRID['C']} × gamma∈{SVM_GRID['gamma']} = {n_svm} config |",
        f"| MERT head | linear/lr1e-3/wd1e-4/drop0.3 (하드코딩, 탐색 0) | arch{MERT_GRID['arch']} × lr{MERT_GRID['lr']} × wd{MERT_GRID['wd']} × drop{MERT_GRID['dropout']} = {n_mert} config |",
        "",
        "> MERT에 SVM보다 큰 탐색 예산을 부여함 (24 vs 12). 모든 탐색은 val_select에서만.",
        "",
        "## 성능 비교 (fold-level 평균±표준편차, n=4)",
        "",
        "| seed | SVM(기존) | SVM(튜닝) | MERT(기존) | MERT(튜닝) | 튜닝MERT − 튜닝SVM | SVM(튜닝, seed_core) | 튜닝MERT − 그것 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdict_rows = []
    for ratio in sorted(df['ratio'].unique()):
        sd_m, sd_s = fold_stats(df, ratio, 'svm_default')
        st_m, st_s = fold_stats(df, ratio, 'svm_tuned')
        md_m, md_s = fold_stats(df, ratio, 'mert_default')
        mt_m, mt_s = fold_stats(df, ratio, 'mert_tuned')
        sf_m, sf_s = fold_stats(df, ratio, 'svm_tuned_full')
        gap, gap_full = mt_m - st_m, mt_m - sf_m
        verdict_rows.append((ratio, mt_m, st_m, gap, sf_m, gap_full))
        lines.append(
            f"| {int(ratio*100)}% | {sd_m:.4f}±{sd_s:.4f} | {st_m:.4f}±{st_s:.4f} "
            f"| {md_m:.4f}±{md_s:.4f} | {mt_m:.4f}±{mt_s:.4f} | **{gap:+.4f}** "
            f"| {sf_m:.4f}±{sf_s:.4f} | **{gap_full:+.4f}** |"
        )

    # 전/후 비교 (A1 재실행 영향)
    before_csv = ROOT / 'results' / 'phase1' / 'mert_tuning_audit_before_A1.csv'
    if before_csv.exists():
        bdf = pd.read_csv(before_csv)
        lines += [
            "",
            "## A1 전/후 비교 — val 분리가 수치를 얼마나 바꿨나",
            "",
            "> 이전 판의 MERT(튜닝)는 early-stop과 config 선택에 같은 val을 써서 낙관 편향이 있었다.",
            "> SVM(튜닝) 열은 이전 판이 labeled의 80%로 학습했으므로, 학습량이 같은",
            "> `SVM(튜닝, seed_core)`와 비교해야 공정하다.",
            "",
            "| seed | MERT(튜닝) 이전 | MERT(튜닝) A1 | 변화 | SVM(튜닝) 이전 | SVM(튜닝, seed_core) A1 | 변화 |",
            "|---|---|---|---|---|---|---|",
        ]
        for ratio in sorted(df['ratio'].unique()):
            bm, _ = fold_stats(bdf, ratio, 'mert_tuned')
            bs, _ = fold_stats(bdf, ratio, 'svm_tuned')
            am, _ = fold_stats(df, ratio, 'mert_tuned')
            as_, _ = fold_stats(df, ratio, 'svm_tuned_full')
            lines.append(f"| {int(ratio*100)}% | {bm:.4f} | {am:.4f} | **{am-bm:+.4f}** "
                         f"| {bs:.4f} | {as_:.4f} | **{as_-bs:+.4f}** |")

    # 판정
    caught_up = [r for r in verdict_rows if r[3] >= -0.005]  # 튜닝 MERT가 튜닝 SVM에 근접/추월
    lines += [
        "",
        "## 판정",
        "",
    ]
    if caught_up:
        lines.append("⚠ **일부 구간에서 튜닝 MERT가 튜닝 SVM에 근접/추월** — 반례 결론 재검토 필요:")
        for ratio, mt, st, gap, sf, gap_f in caught_up:
            lines.append(f"- seed {int(ratio*100)}%: 튜닝 MERT {mt:.4f} vs 튜닝 SVM {st:.4f} ({gap:+.4f}) "
                         f"· seed_core 조건 {sf:.4f} ({gap_f:+.4f})")
    else:
        lines.append("✅ **모든 구간에서 튜닝 SVM > 튜닝 MERT.** "
                     "MERT에 더 큰 val 탐색 예산을 줘도 SVM을 넘지 못함 → 반례 결론 강화.")
        for ratio, mt, st, gap, sf, gap_f in verdict_rows:
            lines.append(f"- seed {int(ratio*100)}%: 튜닝 SVM {st:.4f} vs 튜닝 MERT {mt:.4f} ({gap:+.4f}) "
                         f"· seed_core 조건 {sf:.4f} ({gap_f:+.4f})")

    lines += [
        "",
        "## MERT 탐색이 고른 config (참고 — 언더튜닝 아님을 입증)",
        "",
        "| seed | fold | trial | 최적 MERT config |",
        "|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        if 'mert_best_cfg' in r and isinstance(r['mert_best_cfg'], dict):
            lines.append(f"| {int(r['ratio']*100)}% | {int(r['fold'])} | {int(r['trial'])} | {r['mert_best_cfg']} |")

    OUT.write_text("\n".join(lines) + "\n", encoding='utf-8')
    df.to_csv(ROOT / 'results' / 'phase1' / 'mert_tuning_audit.csv', index=False)
    print(f"\n저장: {OUT}")
    print(f"총 {time.time()-t0:.0f}s")

    # 콘솔 판정 요약
    print("\n=== 판정 ===")
    for ratio, mt, st, gap, sf, gap_f in verdict_rows:
        flag = "⚠ MERT 근접/추월" if gap >= -0.005 else "SVM 우위"
        print(f"  seed {int(ratio*100)}%: 튜닝SVM {st:.4f} vs 튜닝MERT {mt:.4f} ({gap:+.4f})  [{flag}]"
              f"  | seed_core SVM {sf:.4f} ({gap_f:+.4f})")


if __name__ == '__main__':
    main()
