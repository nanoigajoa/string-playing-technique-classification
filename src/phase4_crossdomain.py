"""
Phase 4 — Philharmonia 교차 도메인 검증 + Self-Training 확장

실행:
    python src/phase4_crossdomain.py --step embed    # 단계 4-2: Philharmonia MERT 임베딩 추출
    python src/phase4_crossdomain.py --step eval     # 단계 4-3: Pool B zero-shot 평가 (예정)
    python src/phase4_crossdomain.py --step train    # 단계 4-4: pseudo-labeling + self-training (예정)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# ── 상수 ──────────────────────────────────────────────────────────────────────

CLASSES   = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
STR2LABEL = {v: k for k, v in enumerate(CLASSES)}
LABEL2STR = dict(enumerate(CLASSES))

MERT_MODEL_ID = 'm-a-p/MERT-v1-95M'
MERT_SR       = 24000   # Phase 1과 동일

POOL_A_PATH = ROOT / 'data' / 'philharmonia_pool_a.json'
POOL_B_PATH = ROOT / 'data' / 'philharmonia_pool_b.json'

EMB_PATH      = ROOT / 'data' / 'philharmonia_mert_embeddings.npy'
LAB_PATH      = ROOT / 'data' / 'philharmonia_mert_labels.npy'
META_PATH     = ROOT / 'data' / 'philharmonia_mert_meta.csv'
CHECKPOINT_DIR = ROOT / 'data' / '.philharmonia_embed_checkpoint'

RESULTS_DIR = ROOT / 'results' / 'phase4'


# ══════════════════════════════════════════════════════════════════════════════
# 단계 4-2: Philharmonia MERT 임베딩 추출
# ══════════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    """MPS 사용 가능 여부를 실제 tensor 연산으로 확인 후 디바이스 반환."""
    import torch
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps') + 1
            return 'mps'
        except Exception:
            pass
    return 'cpu'


def _load_mert(device: str):
    """MERT-v1-95M 모델과 프로세서를 로드한다. Phase 1과 동일 설정."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    print(f"  모델 로드 중: {MERT_MODEL_ID}  (device={device})")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MERT_MODEL_ID, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MERT_MODEL_ID, trust_remote_code=True
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  파라미터: {n_params:.1f}M | dtype: {next(model.parameters()).dtype}")
    return processor, model


def _extract_one(filepath: str, processor, model, device: str) -> np.ndarray:
    """오디오 파일 하나를 MERT에 통과시켜 768차원 임베딩을 반환한다."""
    import torch
    import librosa

    y, _ = librosa.load(filepath, sr=MERT_SR, mono=True)
    inputs = processor(y, sampling_rate=MERT_SR, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # last_hidden_state: (1, time_frames, 768) → mean pool → (768,)
    embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().float().numpy()
    return embedding


def _save_checkpoint(embeddings: list, df_done: pd.DataFrame) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    np.save(CHECKPOINT_DIR / 'partial_embeddings.npy', np.array(embeddings, dtype=np.float32))
    df_done.to_csv(CHECKPOINT_DIR / 'partial_meta.csv', index=False)


def _load_checkpoint() -> tuple:
    emb_file  = CHECKPOINT_DIR / 'partial_embeddings.npy'
    meta_file = CHECKPOINT_DIR / 'partial_meta.csv'
    if emb_file.exists() and meta_file.exists():
        return list(np.load(emb_file)), pd.read_csv(meta_file)
    return [], None


def run_embed(force: bool = False) -> None:
    from tqdm import tqdm

    print("=== Phase 4 — 단계 4-2: Philharmonia MERT 임베딩 추출 ===\n")

    if EMB_PATH.exists() and LAB_PATH.exists() and META_PATH.exists() and not force:
        existing = np.load(EMB_PATH).shape
        print(f"임베딩 캐시 이미 존재: {EMB_PATH}  shape={existing}")
        print("재추출하려면 --force 옵션을 사용하세요.")
        return

    assert POOL_A_PATH.exists() and POOL_B_PATH.exists(), (
        "Pool A/B JSON 없음. 먼저 실행:\n"
        "  python data/build_philharmonia_splits.py"
    )

    # Pool A + Pool B 전체 파일 목록 구성
    with open(POOL_A_PATH, encoding='utf-8') as f:
        pool_a = json.load(f)
    with open(POOL_B_PATH, encoding='utf-8') as f:
        pool_b = json.load(f)

    all_samples = [dict(s, pool='A') for s in pool_a] + \
                  [dict(s, pool='B') for s in pool_b]

    print(f"대상 파일: Pool A {len(pool_a)}개 + Pool B {len(pool_b)}개 = {len(all_samples)}개\n")

    # 디바이스 및 모델 로드
    device = _get_device()
    print(f"디바이스: {device}")
    processor, model = _load_mert(device)

    # 체크포인트 이어 시작
    embeddings, df_done = _load_checkpoint()
    done_paths = set(df_done['path'].tolist()) if df_done is not None else set()
    if done_paths:
        print(f"\n체크포인트 발견: {len(done_paths)}개 이어서 시작")

    todo      = [s for s in all_samples if s['path'] not in done_paths]
    meta_rows = [] if df_done is None else df_done.to_dict('records')
    errors    = []
    CHECKPOINT_EVERY = 50

    print(f"\n임베딩 추출 시작: {len(todo)}개 남음\n")
    for i, sample in enumerate(tqdm(todo, desc='MERT 추출')):
        filepath = str(ROOT / sample['path'])
        try:
            emb = _extract_one(filepath, processor, model, device)
            embeddings.append(emb)
            meta_rows.append({
                'path':       sample['path'],
                'label':      sample['label'],
                'label_id':   STR2LABEL[sample['label']],
                'instrument': sample['instrument'],
                'pool':       sample['pool'],
            })
        except Exception as e:
            errors.append((sample['path'], str(e)))
            tqdm.write(f"  [오류] {sample['path']}: {e}")
            continue

        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(embeddings, pd.DataFrame(meta_rows))
            tqdm.write(f"  체크포인트: {len(embeddings)}개 완료")

    # 최종 저장
    print(f"\n성공: {len(embeddings)} / 실패: {len(errors)}")

    emb_array = np.array(embeddings, dtype=np.float32)  # (N, 768)
    df_meta   = pd.DataFrame(meta_rows)
    lab_array = df_meta['label_id'].to_numpy(dtype=np.int32)

    EMB_PATH.parent.mkdir(exist_ok=True)
    np.save(EMB_PATH, emb_array)
    np.save(LAB_PATH, lab_array)
    df_meta.to_csv(META_PATH, index=False)

    print(f"\n저장 완료:")
    print(f"  {EMB_PATH}   shape={emb_array.shape}")
    print(f"  {LAB_PATH}   shape={lab_array.shape}")
    print(f"  {META_PATH}  rows={len(df_meta)}")

    pool_counts = df_meta['pool'].value_counts()
    print(f"\nPool 분포: A={pool_counts.get('A', 0)}, B={pool_counts.get('B', 0)}")

    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.iterdir():
            f.unlink()
        CHECKPOINT_DIR.rmdir()


# ══════════════════════════════════════════════════════════════════════════════
# 단계 4-3: Pool B Zero-Shot 평가
# ══════════════════════════════════════════════════════════════════════════════

def run_eval() -> None:
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score)

    print("=== Phase 4 — 단계 4-3: Pool B Zero-Shot 평가 ===\n")

    # 사전 조건 확인
    orch_emb_path = ROOT / 'data' / 'mert_embeddings.npy'
    orch_lab_path = ROOT / 'data' / 'mert_labels.npy'
    for p in [EMB_PATH, META_PATH, orch_emb_path, orch_lab_path]:
        assert p.exists(), f"파일 없음: {p}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── OrchideaSOL로 MERT head 학습 ─────────────────────────────────────────
    print("1) OrchideaSOL MERT head 학습 (전체 라벨 사용)...")
    X_orch = np.load(orch_emb_path).astype(np.float32)
    y_orch = np.load(orch_lab_path).astype(np.int32)
    print(f"   OrchideaSOL 임베딩: {X_orch.shape}  클래스: {np.unique(y_orch)}")

    head = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                              class_weight='balanced')
    head.fit(X_orch, y_orch)

    # ── 정직한 in-domain 기준 = LOSO (악기 단위 GroupKFold) ────────────────────
    # A4 수정: 이전에는 in-sample 재예측(0.9951)을 도메인 손실 기준으로 썼으나,
    # 그 값은 학습셋을 그대로 맞힌 낙관 수치라 cross-domain 손실을 부풀렸다.
    # 도메인 이동 손실은 "보지 못한 악기로의 일반화(LOSO)"와 비교해야 공정하다.
    from sklearn.model_selection import GroupKFold, cross_val_predict
    groups_orch = pd.read_csv(ROOT / 'data' / 'features_full.csv')['instrument'].values
    assert len(groups_orch) == len(y_orch), "features_full 행 정렬 불일치"
    gkf = GroupKFold(n_splits=len(np.unique(groups_orch)))
    y_orch_loso = cross_val_predict(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        X_orch, y_orch, groups=groups_orch, cv=gkf)
    f1_indomain = f1_score(y_orch, y_orch_loso, average='macro')          # LOSO — 정직한 기준
    f1_insample = f1_score(y_orch, head.predict(X_orch), average='macro')  # in-sample — 착시(참고만)
    print(f"   OrchideaSOL in-domain LOSO Macro F1: {f1_indomain:.4f} "
          f"(in-sample {f1_insample:.4f} — 낙관 착시, 기준 아님)")

    # ── Pool B 로드 ──────────────────────────────────────────────────────────
    print("\n2) Pool B 임베딩 로드...")
    df_meta = pd.read_csv(META_PATH)
    mask_b  = df_meta['pool'] == 'B'
    X_b     = np.load(EMB_PATH).astype(np.float32)[mask_b.values]
    y_b     = np.load(LAB_PATH).astype(np.int32)[mask_b.values]
    print(f"   Pool B: {X_b.shape[0]}개")
    for cls_id, cls_name in LABEL2STR.items():
        n = int((y_b == cls_id).sum())
        print(f"     {cls_name:<22} {n:>4}")

    # ── MERT head zero-shot 평가 ─────────────────────────────────────────────
    print("\n3) MERT head → Pool B 예측...")
    y_pred_mert = head.predict(X_b)
    f1_mert = f1_score(y_b, y_pred_mert, average='macro')
    print(f"   Pool B Macro F1 (MERT head): {f1_mert:.4f}")
    print(f"   도메인 이동 손실 (LOSO 대비): {f1_indomain - f1_mert:+.4f}")

    report_mert = classification_report(
        y_b, y_pred_mert,
        target_names=CLASSES, digits=3, zero_division=0
    )
    print("\n   Per-class 결과 (MERT head):")
    print(report_mert)

    # ── SVM v1 zero-shot 평가 ────────────────────────────────────────────────
    svm_path = ROOT / 'models' / 'svm_v1.joblib'
    if svm_path.exists():
        print("4) SVM v1 → Pool B 예측 (features.py 필요)...")
        print("   [건너뜀] SVM은 features.py 특징 추출이 필요 — 별도 실행 필요")
    else:
        print("4) SVM v1 모델 없음 — 건너뜀")

    # ── Confusion Matrix 저장 ────────────────────────────────────────────────
    cm = confusion_matrix(y_b, y_pred_mert)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
        xticklabels=[c.replace('_', '\n') for c in CLASSES],
        yticklabels=CLASSES,
        xlabel='Predicted', ylabel='True',
        title=f'Pool B Zero-Shot (MERT head)\nMacro F1={f1_mert:.3f}',
    )
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=9)
    plt.tight_layout()
    png_path = RESULTS_DIR / 'poolB_zeroshot_confusion.png'
    fig.savefig(png_path, dpi=150)
    plt.close()
    print(f"\n   confusion matrix 저장: {png_path}")

    # ── 리포트 저장 ──────────────────────────────────────────────────────────
    report_md = f"""# Phase 4 — Pool B Zero-Shot 평가 결과

## 모델: MERT head (OrchideaSOL 전체 라벨 학습 후 Philharmonia Pool B 적용)

| 지표 | 값 |
|---|---|
| OrchideaSOL in-domain **LOSO** Macro F1 (기준) | {f1_indomain:.4f} |
| **Pool B Cross-Domain Macro F1** | **{f1_mert:.4f}** |
| **도메인 이동 손실 (LOSO 대비)** | **{f1_indomain - f1_mert:+.4f}** |
| — in-sample 재예측 (낙관 착시, 기준 아님) | {f1_insample:.4f} |

> **A4 수정**: 도메인 손실은 in-domain **LOSO**(보지 못한 악기로의 일반화)를 기준으로 잰다.
> 이전 판은 in-sample 재예측 {f1_insample:.4f}를 기준으로 써 손실을 {f1_insample - f1_mert:+.4f}로
> 부풀렸다. 정직한 손실은 **{f1_indomain - f1_mert:+.4f}**이다.

## Per-class F1 (Pool B)

```
{report_mert}
```

## Pool B 구성

| 클래스 | 샘플 수 |
|---|---|
""" + "\n".join(
        f"| {LABEL2STR[i]} | {int((y_b == i).sum())} |"
        for i in range(len(CLASSES))
    ) + f"""

## 시각화

- `poolB_zeroshot_confusion.png`

---
*생성일: Phase 4 단계 4-3*
"""
    report_path = RESULTS_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"   리포트 저장: {report_path}")

    print(f"\n=== 완료: Pool B Cross-Domain Macro F1 = {f1_mert:.4f} ===")


# ══════════════════════════════════════════════════════════════════════════════
# 단계 4-4: Pseudo-Labeling + Self-Training
# ══════════════════════════════════════════════════════════════════════════════

TAU_LIST   = [0.7, 0.8, 0.9]
MAX_ITER   = 2
ORDINARIO_PSEUDO_CAP = 150  # pseudo-label 단계에서도 ordinario 쏠림 방지

def _train_head(X: np.ndarray, y: np.ndarray) -> object:
    from sklearn.linear_model import LogisticRegression
    head = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                              class_weight='balanced')
    head.fit(X, y)
    return head


def _pseudo_label(head, X_pool: np.ndarray, tau: float,
                  y_true_pool: np.ndarray) -> dict:
    """
    Pool A에 pseudo-label을 붙이고 정확도를 측정한다.
    반환: {adopted_X, adopted_y, accuracy, per_class_stats}
    """
    proba  = head.predict_proba(X_pool)          # (N, 5)
    conf   = proba.max(axis=1)                    # max confidence per sample
    pred   = proba.argmax(axis=1)

    mask   = conf >= tau
    adopted_X = X_pool[mask]
    adopted_y = pred[mask]
    true_y    = y_true_pool[mask]

    accuracy = float((adopted_y == true_y).mean()) if mask.sum() > 0 else 0.0

    per_class = {}
    for cls_id, cls_name in LABEL2STR.items():
        cls_mask = adopted_y == cls_id
        n_adopted = int(cls_mask.sum())
        n_correct = int((adopted_y[cls_mask] == true_y[cls_mask]).sum()) if n_adopted > 0 else 0
        per_class[cls_name] = {'adopted': n_adopted, 'correct': n_correct}

    # ordinario pseudo-label 상한 적용
    ord_id   = CLASSES.index('ordinario')
    ord_mask = adopted_y == ord_id
    if ord_mask.sum() > ORDINARIO_PSEUDO_CAP:
        ord_indices = np.where(ord_mask)[0]
        # confidence 높은 순으로 상한만 유지
        ord_conf = conf[mask][ord_mask]
        keep = ord_indices[np.argsort(ord_conf)[::-1][:ORDINARIO_PSEUDO_CAP]]
        non_ord_indices = np.where(~ord_mask)[0]
        keep_all = np.sort(np.concatenate([keep, non_ord_indices]))
        adopted_X = adopted_X[keep_all]
        adopted_y = adopted_y[keep_all]
        per_class['ordinario']['adopted'] = ORDINARIO_PSEUDO_CAP

    return {
        'adopted_X':    adopted_X,
        'adopted_y':    adopted_y,
        'n_total':      len(X_pool),
        'n_adopted':    int(mask.sum()),
        'accuracy':     accuracy,
        'per_class':    per_class,
    }


def _select_tau_on_source(X_orch, y_orch, X_a, y_a_true):
    """τ를 OrchideaSOL 내부 val에서 선택 (타깃 도메인 라벨 미사용, 결함 B 수정).

    각 τ로 Pool A를 pseudo-label해 OrchideaSOL-train에 추가한 뒤
    OrchideaSOL-val Macro F1을 측정 → 최고 τ 반환.
    Philharmonia(Pool B) 라벨은 전혀 참조하지 않는다.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    Xtr, Xval, ytr, yval = train_test_split(
        X_orch, y_orch, test_size=0.2, stratify=y_orch, random_state=42
    )
    tau_scores = {}
    for tau in TAU_LIST:
        X_train, y_train = Xtr.copy(), ytr.copy()
        head = _train_head(X_train, y_train)
        for _ in range(MAX_ITER):
            pl = _pseudo_label(head, X_a, tau, y_a_true)
            if pl['n_adopted'] > 0:
                X_train = np.concatenate([Xtr, pl['adopted_X']])
                y_train = np.concatenate([ytr, pl['adopted_y']])
            head = _train_head(X_train, y_train)
        tau_scores[tau] = float(f1_score(yval, head.predict(Xval), average='macro'))
    best_tau = max(tau_scores, key=tau_scores.get)
    return best_tau, tau_scores


def run_train() -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    print("=== Phase 4 — 단계 4-4: Pseudo-Labeling + Self-Training ===\n")

    orch_emb_path = ROOT / 'data' / 'mert_embeddings.npy'
    orch_lab_path = ROOT / 'data' / 'mert_labels.npy'
    for p in [EMB_PATH, META_PATH, orch_emb_path, orch_lab_path]:
        assert p.exists(), f"파일 없음: {p}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    X_orch = np.load(orch_emb_path).astype(np.float32)
    y_orch = np.load(orch_lab_path).astype(np.int32)

    df_meta  = pd.read_csv(META_PATH)
    emb_all  = np.load(EMB_PATH).astype(np.float32)
    lab_all  = np.load(LAB_PATH).astype(np.int32)

    mask_a = (df_meta['pool'] == 'A').values
    mask_b = (df_meta['pool'] == 'B').values
    X_a, y_a_true = emb_all[mask_a], lab_all[mask_a]   # Pool A (true label 보유)
    X_b, y_b      = emb_all[mask_b], lab_all[mask_b]   # Pool B (holdout)

    print(f"OrchideaSOL: {X_orch.shape[0]}개  "
          f"Pool A: {X_a.shape[0]}개  Pool B: {X_b.shape[0]}개\n")

    # ── τ 선택: OrchideaSOL 내부 val (타깃 도메인 라벨 미사용, 결함 B 수정) ─────
    #    기존: best_tau = Pool B(holdout) F1 최대 → 누수. 제거함.
    best_tau, tau_scores = _select_tau_on_source(X_orch, y_orch, X_a, y_a_true)
    print(f"{'='*55}")
    print("τ 선택 — OrchideaSOL 내부 val Macro F1 (타깃 정보 미사용):")
    for tau in TAU_LIST:
        mark = "  ← 선택" if tau == best_tau else ""
        print(f"  τ={tau}: {tau_scores[tau]:.4f}{mark}")
    print(f"선택된 τ = {best_tau}\n")

    # ── 선택된 τ로 self-training (adoption 통계 기록; Pool B는 여기서 평가 안 함) ─
    summary_rows = []
    X_train_best = X_orch.copy()
    y_train_best = y_orch.copy()
    head_best = _train_head(X_train_best, y_train_best)
    for iteration in range(MAX_ITER):
        pl = _pseudo_label(head_best, X_a, best_tau, y_a_true)
        adopt_rate = pl['n_adopted'] / pl['n_total']
        print(f"  Iter {iteration+1}: 채택 {pl['n_adopted']}/{pl['n_total']} "
              f"({adopt_rate:.1%}) | pseudo 정확도 {pl['accuracy']:.4f}")
        summary_rows.append({
            'tau':        best_tau,
            'iteration':  iteration + 1,
            'n_adopted':  pl['n_adopted'],
            'adopt_rate': round(adopt_rate, 4),
            'pseudo_acc': round(pl['accuracy'], 4),
        })
        if pl['n_adopted'] > 0:
            X_train_best = np.concatenate([X_orch, pl['adopted_X']])
            y_train_best = np.concatenate([y_orch, pl['adopted_y']])
        head_best = _train_head(X_train_best, y_train_best)

    df_summary = pd.DataFrame(summary_rows)

    # ── Pool B 최종 평가 (선택이 아닌 보고 목적, 1회만) ──────────────────────────
    y_pred_best = head_best.predict(X_b)
    f1_best = f1_score(y_b, y_pred_best, average='macro')
    report_best = classification_report(y_b, y_pred_best,
                                        target_names=CLASSES, digits=3,
                                        zero_division=0)
    print(f"\nPool B Macro F1 (τ={best_tau}, 최종 1회 평가): {f1_best:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_b, y_pred_best)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
        xticklabels=[c.replace('_', '\n') for c in CLASSES],
        yticklabels=CLASSES,
        xlabel='Predicted', ylabel='True',
        title=f'Pool B After Self-Training (τ={best_tau})\nMacro F1={f1_best:.3f}',
    )
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=9)
    plt.tight_layout()
    png_path = RESULTS_DIR / 'poolB_selftraining_confusion.png'
    fig.savefig(png_path, dpi=150)
    plt.close()

    # Pseudo-label 정확도 막대 그래프 (best τ, iteration 1)
    pl_stats = _pseudo_label(
        _train_head(X_orch, y_orch), X_a, best_tau, y_a_true
    )
    cls_names = list(pl_stats['per_class'].keys())
    cls_accs  = [
        (s['correct'] / s['adopted'] if s['adopted'] > 0 else 0.0)
        for s in pl_stats['per_class'].values()
    ]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(cls_names, cls_accs, color='steelblue')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Pseudo-label 정확도')
    ax2.set_title(f'Pseudo-Label 클래스별 정확도 (τ={best_tau})')
    ax2.tick_params(axis='x', rotation=20)
    for bar, acc in zip(bars, cls_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'pseudolabel_accuracy_by_class.png', dpi=150)
    plt.close()

    # 리포트 업데이트
    zero_shot_f1 = 0.4226  # run_eval 결과 (Pool B zero-shot)
    tau_sel_table = "\n".join(
        f"| {tau} | {tau_scores[tau]:.4f} | {'← 선택' if tau == best_tau else ''} |"
        for tau in TAU_LIST
    )
    adopt_table = "\n".join(
        f"| {r['iteration']} | {r['n_adopted']} | {r['adopt_rate']:.1%} | {r['pseudo_acc']:.4f} |"
        for _, r in df_summary.iterrows()
    )
    report_md = f"""# Phase 4 — Self-Training 결과 (누수 수정본)

## τ 선택 — OrchideaSOL 내부 val (타깃 도메인 라벨 미사용)

> 결함 B 수정: Pool B(holdout)로 τ를 고르던 누수를 제거.
> τ는 소스 도메인(OrchideaSOL) val에서만 결정하고 Philharmonia에 그대로 적용.
> "타깃 도메인 정보 없이 정한 τ가 새 도메인에서 작동하는가"를 측정한다.

| τ | OrchideaSOL val Macro F1 | |
|---|---|---|
{tau_sel_table}

선택된 τ = **{best_tau}**

## 선택된 τ에서 pseudo-label 채택 (Pool A)

| Iter | 채택 수 | 채택률 | Pseudo-label 정확도 |
|---|---|---|---|
{adopt_table}

## 비교 (Pool B — 최종 1회 평가)

| 조건 | Pool B Macro F1 |
|---|---|
| Zero-shot 기준선 (단계 4-3) | {zero_shot_f1:.4f} |
| **Self-training (τ={best_tau}, 소스 val에서 선택)** | **{f1_best:.4f}** |
| 향상폭 | {f1_best - zero_shot_f1:+.4f} |

## Per-class (Self-training τ={best_tau})

```
{report_best}
```

## 시각화

- `poolB_selftraining_confusion.png`
- `pseudolabel_accuracy_by_class.png`
"""
    with open(RESULTS_DIR / 'report.md', 'a', encoding='utf-8') as f:
        f.write("\n\n---\n\n" + report_md)

    print(f"\n저장 완료: {RESULTS_DIR}")
    print(f"Zero-shot → Self-training: {zero_shot_f1:.4f} → {f1_best:.4f} "
          f"({f1_best - zero_shot_f1:+.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 4 — Cross-Domain 파이프라인')
    parser.add_argument('--step', choices=['embed', 'eval', 'train'], default='embed',
                        help='실행할 단계 (기본값: embed)')
    parser.add_argument('--force', action='store_true',
                        help='캐시 무시하고 재실행')
    args = parser.parse_args()

    if args.step == 'embed':
        run_embed(force=args.force)
    elif args.step == 'eval':
        run_eval()
    elif args.step == 'train':
        run_train()


if __name__ == '__main__':
    main()
