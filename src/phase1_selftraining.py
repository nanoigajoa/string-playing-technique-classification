"""
Phase 1 — Self-Training 파이프라인

실행:
    python src/phase1_selftraining.py --step splits   # 1-1: 분할 설계
    python src/phase1_selftraining.py --step embed    # 1-2: MERT 임베딩 추출
    python src/phase1_selftraining.py                 # 기본값: splits
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

def parse_recording_id(filename: str) -> str:
    """파일명에서 녹음 ID 추출 (features.py와 동일 로직, librosa 의존 없음)
    Vn-ord-A#3-ff-4c-N.wav → Vn-A#3-ff
    """
    parts = Path(filename).stem.split('-')
    if len(parts) >= 4:
        return f'{parts[0]}-{parts[2]}-{parts[3]}'
    return parts[0]


# ── 공통 상수 ─────────────────────────────────────────────────────────────────

CLASSES     = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
LABEL2STR   = dict(enumerate(CLASSES))
STR2LABEL   = {v: k for k, v in LABEL2STR.items()}
INSTRUMENTS = ['Violin', 'Viola', 'Violoncello', 'Contrabass']

# 1-1 관련
DATA_PATH      = ROOT / 'data' / 'features_full.csv'
SPLITS_PATH    = ROOT / 'data' / 'split_configs.json'
SEED_RATIOS    = [0.05, 0.10, 0.20, 0.50]
N_TRIALS       = 3
BASE_SEED      = 42

# 1-2 관련
MERT_MODEL_ID  = 'm-a-p/MERT-v1-95M'
MERT_SR        = 24000   # MERT 요구 샘플링 레이트
DATA_ROOT      = ROOT / '_OrchideaSOL2020_release' / 'OrchideaSOL2020' / 'Strings'
EMB_PATH       = ROOT / 'data' / 'mert_embeddings.npy'
LAB_PATH       = ROOT / 'data' / 'mert_labels.npy'
META_PATH      = ROOT / 'data' / 'mert_meta.csv'
CHECKPOINT_DIR = ROOT / 'data' / '.embed_checkpoint'


# ══════════════════════════════════════════════════════════════════════════════
# 단계 1-1: 데이터 분할 설계
# ══════════════════════════════════════════════════════════════════════════════

def load_meta() -> pd.DataFrame:
    """features_full.csv에서 메타 컬럼만 로드하고 0-based 정수 인덱스로 재설정."""
    df = pd.read_csv(DATA_PATH, usecols=['label_id', 'instrument'])
    df['label'] = df['label_id'].map(LABEL2STR)
    assert df['label'].notna().all(), "label_id → label 매핑 실패: 알 수 없는 label_id 존재"
    return df.reset_index(drop=True)


def build_splits(df: pd.DataFrame) -> dict:
    """
    LOSO × seed ratio × trial 조합으로 labeled / unlabeled / test 인덱스를 생성한다.

    주의: features_full.csv에 recording_id 컬럼이 없으므로
          labeled seed 분리는 클래스 단위 stratified sampling으로 수행한다.
          1-2 완료 후 mert_meta.csv에 recording_id가 생기면 더 세밀한 분리 가능.
    """
    instruments = sorted(df['instrument'].unique().tolist())
    gkf = GroupKFold(n_splits=len(instruments))

    splits = []
    for fold_idx, (train_global, test_global) in enumerate(
        gkf.split(df, groups=df['instrument'])
    ):
        test_instrument = df['instrument'].iloc[test_global[0]]
        df_train = df.iloc[train_global].reset_index(drop=True)

        for ratio in SEED_RATIOS:
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(BASE_SEED + trial * 100 + int(ratio * 1000))

                labeled_local   = _stratified_sample(df_train, ratio, rng)
                unlabeled_local = np.setdiff1d(np.arange(len(df_train)), labeled_local)

                labeled_idx   = train_global[labeled_local].tolist()
                unlabeled_idx = train_global[unlabeled_local].tolist()
                test_idx      = test_global.tolist()

                splits.append({
                    'fold':                  fold_idx,
                    'test_instrument':       test_instrument,
                    'seed_ratio':            ratio,
                    'trial':                 trial,
                    'n_labeled':             len(labeled_idx),
                    'n_unlabeled':           len(unlabeled_idx),
                    'n_test':                len(test_idx),
                    'labeled_class_dist':    _class_dist(df, labeled_idx),
                    'unlabeled_class_dist':  _class_dist(df, unlabeled_idx),
                    'labeled_idx':           labeled_idx,
                    'unlabeled_idx':         unlabeled_idx,
                    'test_idx':              test_idx,
                })

    return {
        'metadata': {
            'total_samples': len(df),
            'instruments':   instruments,
            'seed_ratios':   SEED_RATIOS,
            'n_trials':      N_TRIALS,
            'n_folds':       len(instruments),
            'group_key':     'instrument',
            'classes':       CLASSES,
        },
        'splits': splits,
    }


def _stratified_sample(df_train: pd.DataFrame, ratio: float, rng) -> np.ndarray:
    selected = []
    for cls in CLASSES:
        positions = np.where(df_train['label'].values == cls)[0]
        n_select  = max(1, round(len(positions) * ratio))
        chosen    = rng.choice(positions, size=n_select, replace=False)
        selected.append(chosen)
    return np.concatenate(selected)


def _class_dist(df: pd.DataFrame, idx: list) -> dict:
    counts = df['label'].iloc[idx].value_counts()
    return {cls: int(counts.get(cls, 0)) for cls in CLASSES}


def validate_splits(config: dict, df: pd.DataFrame) -> None:
    errors = []
    for s in config['splits']:
        labeled, unlabeled, test = set(s['labeled_idx']), set(s['unlabeled_idx']), set(s['test_idx'])
        if labeled & unlabeled:
            errors.append(f"fold={s['fold']} ratio={s['seed_ratio']}: labeled ∩ unlabeled 겹침")
        if (labeled | unlabeled) & test:
            errors.append(f"fold={s['fold']} ratio={s['seed_ratio']}: train ∩ test 겹침")
        if len(labeled) + len(unlabeled) + len(test) != len(df):
            errors.append(f"fold={s['fold']} ratio={s['seed_ratio']}: 합계 불일치")
    if errors:
        for e in errors:
            print(f"  [오류] {e}")
        raise AssertionError(f"검증 실패 {len(errors)}개")
    print(f"  검증 통과: {len(config['splits'])}개 split 모두 정상")


def print_splits_summary(config: dict) -> None:
    meta = config['metadata']
    print(f"\n총 샘플: {meta['total_samples']} | "
          f"Folds: {meta['n_folds']} | "
          f"Ratios: {meta['seed_ratios']} | "
          f"Trials: {meta['n_trials']} | "
          f"생성: {len(config['splits'])}개\n")
    header = f"{'fold':<5} {'test_inst':<14} {'ratio':>6} {'trial':>6} {'labeled':>8} {'unlabeled':>10} {'test':>6}"
    print(header)
    print('-' * len(header))
    prev = None
    for s in config['splits']:
        if s['fold'] != prev:
            if prev is not None:
                print()
            prev = s['fold']
        print(f"{s['fold']:<5} {s['test_instrument']:<14} {s['seed_ratio']:>6.0%} "
              f"{s['trial']:>6} {s['n_labeled']:>8} {s['n_unlabeled']:>10} {s['n_test']:>6}")


def run_build_splits():
    assert DATA_PATH.exists(), f"features_full.csv 없음: {DATA_PATH}"
    print("=== Phase 1 — 단계 1-1: 데이터 분할 설계 ===\n")
    df = load_meta()
    print(f"데이터 로드: {df.shape}")
    for cls, cnt in df['label'].value_counts().items():
        print(f"  {cls:<22} {cnt:>5}")

    config = build_splits(df)
    print("\n검증 중...")
    validate_splits(config, df)
    print_splits_summary(config)

    SPLITS_PATH.parent.mkdir(exist_ok=True)
    with open(SPLITS_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {SPLITS_PATH}  ({SPLITS_PATH.stat().st_size / 1024:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 단계 1-2: MERT 임베딩 추출
# ══════════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    """MPS 사용 가능 여부를 실제 tensor 연산으로 확인 후 디바이스 반환."""
    import torch
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps') + 1  # 실제 op 테스트
            return 'mps'
        except Exception:
            pass
    return 'cpu'


def _load_mert(device: str):
    """MERT-v1-95M 모델과 프로세서를 로드한다."""
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
    """
    오디오 파일 하나를 MERT에 통과시켜 768차원 임베딩을 반환한다.
    last hidden state를 time axis로 mean pooling.
    """
    import torch
    import librosa

    y, _ = librosa.load(filepath, sr=MERT_SR, mono=True)

    inputs = processor(y, sampling_rate=MERT_SR, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # last_hidden_state: (1, time_frames, 768)
    embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().float().numpy()
    return embedding  # (768,)


def _scan_audio_files() -> pd.DataFrame:
    """OrchideaSOL에서 대상 오디오 파일 목록을 수집한다."""
    rows = []
    for instrument in INSTRUMENTS:
        for technique in CLASSES:
            folder = DATA_ROOT / instrument / technique
            if not folder.exists():
                continue
            for f in sorted(folder.glob('*.wav')):
                rows.append({
                    'filepath':     str(f),
                    'filename':     f.name,
                    'label':        technique,
                    'label_id':     STR2LABEL[technique],
                    'instrument':   instrument,
                    'recording_id': parse_recording_id(f.name),
                })
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


def _save_checkpoint(embeddings: list, df_done: pd.DataFrame) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    np.save(CHECKPOINT_DIR / 'partial_embeddings.npy', np.array(embeddings, dtype=np.float32))
    df_done.to_csv(CHECKPOINT_DIR / 'partial_meta.csv', index=False)


def _load_checkpoint() -> tuple:
    """저장된 체크포인트가 있으면 (embeddings_list, meta_df) 반환, 없으면 ([], None)."""
    emb_file  = CHECKPOINT_DIR / 'partial_embeddings.npy'
    meta_file = CHECKPOINT_DIR / 'partial_meta.csv'
    if emb_file.exists() and meta_file.exists():
        embeddings = list(np.load(emb_file))
        meta       = pd.read_csv(meta_file)
        return embeddings, meta
    return [], None


def run_build_embeddings(force: bool = False):
    import torch
    from tqdm import tqdm

    print("=== Phase 1 — 단계 1-2: MERT 임베딩 추출 ===\n")

    # 이미 완료된 경우 스킵
    if EMB_PATH.exists() and LAB_PATH.exists() and META_PATH.exists() and not force:
        existing = np.load(EMB_PATH).shape
        print(f"임베딩 캐시 이미 존재: {EMB_PATH}  shape={existing}")
        print("재추출하려면 --force 옵션을 사용하세요.")
        return

    assert DATA_ROOT.exists(), (
        f"OrchideaSOL 데이터 없음: {DATA_ROOT}\n"
        f"  IRCAM Forum에서 다운로드 후 _OrchideaSOL2020_release/ 에 배치하세요."
    )

    # 오디오 파일 목록 수집
    print("오디오 파일 스캔 중...")
    df_files = _scan_audio_files()
    print(f"  발견: {len(df_files)}개")
    for cls, cnt in df_files['label'].value_counts().items():
        print(f"    {cls:<22} {cnt:>5}")

    # 디바이스 확인
    device = _get_device()
    print(f"\n디바이스: {device}")

    # 모델 로드
    processor, model = _load_mert(device)

    # 체크포인트 로드 (이전에 중단됐다면 이어서 시작)
    embeddings, df_done = _load_checkpoint()
    done_files = set(df_done['filepath'].tolist()) if df_done is not None else set()
    if done_files:
        print(f"\n체크포인트 발견: {len(done_files)}개 이어서 시작")

    df_todo  = df_files[~df_files['filepath'].isin(done_files)].reset_index(drop=True)
    meta_rows = [] if df_done is None else df_done.to_dict('records')

    # 임베딩 추출
    errors = []
    CHECKPOINT_EVERY = 100

    print(f"\n임베딩 추출 시작: {len(df_todo)}개 남음\n")
    for i, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc='MERT 추출'):
        try:
            emb = _extract_one(row['filepath'], processor, model, device)
            embeddings.append(emb)
            meta_rows.append(row.to_dict())
        except Exception as e:
            errors.append((row['filename'], str(e)))
            tqdm.write(f"  [오류] {row['filename']}: {e}")
            continue

        # 체크포인트 저장
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(embeddings, pd.DataFrame(meta_rows))
            tqdm.write(f"  체크포인트 저장: {len(embeddings)}개 완료")

    # 최종 저장
    print(f"\n성공: {len(embeddings)} / 실패: {len(errors)}")

    emb_array = np.array(embeddings, dtype=np.float32)   # (N, 768)
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

    # 체크포인트 정리
    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.iterdir():
            f.unlink()
        CHECKPOINT_DIR.rmdir()
        print("  체크포인트 삭제 완료")

    # 보너스: recording_id가 생겼으므로 split_configs 업데이트 안내
    if SPLITS_PATH.exists():
        print(f"\n참고: {META_PATH}에 recording_id 컬럼이 생겼습니다.")
        print("      split_configs.json을 recording_id 기반으로 재생성하려면")
        print("      python src/phase1_selftraining.py --step splits --use-recording-id")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Phase 1 — Self-Training 파이프라인')
    parser.add_argument(
        '--step',
        choices=['splits', 'embed'],
        default='splits',
        help='실행할 단계 (기본값: splits)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='embed: 기존 캐시가 있어도 재추출',
    )
    args = parser.parse_args()

    if args.step == 'splits':
        run_build_splits()
    elif args.step == 'embed':
        run_build_embeddings(force=args.force)


if __name__ == '__main__':
    main()
