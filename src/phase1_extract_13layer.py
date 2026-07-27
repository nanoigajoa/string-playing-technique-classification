"""
B1 — 13층 임베딩 캐시 재추출 (Track B 선행)

목적: 기존 캐시는 last_hidden_state(마지막 층)만 mean-pool해 관행(레이어 가중 결합)
      이하였다. MERT-v1-95M의 output_hidden_states 전 13층(임베딩 출력 + 12 transformer)
      을 각각 time mean-pool해 (N, 13, 768)로 저장한다. 이후 B2(레이어별 probe·학습형
      가중), B4(역방향 cross-domain), B3(LoRA)이 이 캐시를 재사용한다.

정렬: 기존 라벨/특징과 행 순서를 유지하기 위해 mert_meta.csv(filepath),
      philharmonia_mert_meta.csv(path)를 **파일에 적힌 순서 그대로** 처리한다.
      검증: 마지막 층(index 12) mean-pool이 기존 mert_embeddings.npy와 일치해야 한다.

실행:
    python src/phase1_extract_13layer.py                 # 둘 다
    python src/phase1_extract_13layer.py --only orch      # OrchideaSOL만
    python src/phase1_extract_13layer.py --only phil      # Philharmonia만
    python src/phase1_extract_13layer.py --verify         # 정렬 검증만 (추출 후)

출력:
    data/mert_embeddings_13layer.npy                (3402, 13, 768) float32
    data/philharmonia_mert_embeddings_13layer.npy   (577, 13, 768)  float32
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

MERT_MODEL_ID = 'm-a-p/MERT-v1-95M'
MERT_SR       = 24000
N_LAYERS      = 13          # 임베딩 출력 + 12 transformer 층

SOURCES = {
    'orch': {
        'meta':     ROOT / 'data' / 'mert_meta.csv',
        'path_col': 'filepath',
        'abs':      True,     # filepath는 절대경로
        'out':      ROOT / 'data' / 'mert_embeddings_13layer.npy',
        'ref':      ROOT / 'data' / 'mert_embeddings.npy',   # 마지막 층 검증용
        'ckpt':     ROOT / 'data' / '.embed13_ckpt_orch',
    },
    'phil': {
        'meta':     ROOT / 'data' / 'philharmonia_mert_meta.csv',
        'path_col': 'path',
        'abs':      False,    # path는 ROOT 기준 상대경로
        'out':      ROOT / 'data' / 'philharmonia_mert_embeddings_13layer.npy',
        'ref':      ROOT / 'data' / 'philharmonia_mert_embeddings.npy',
        'ckpt':     ROOT / 'data' / '.embed13_ckpt_phil',
    },
}


def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps') + 1
            return 'mps'
        except Exception:
            pass
    return 'cpu'


def load_mert(device: str):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    print(f"  모델 로드: {MERT_MODEL_ID} (device={device})", flush=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True).to(device).eval()
    return processor, model


def extract_one(filepath: str, processor, model, device: str) -> np.ndarray:
    """오디오 1개 → (13, 768): 각 hidden layer를 time축 mean-pool."""
    import torch
    import librosa
    y, _ = librosa.load(filepath, sr=MERT_SR, mono=True)
    inputs = processor(y, sampling_rate=MERT_SR, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple(len=13) of (1, T, 768) → stack → (13, T, 768) → mean over T
    hs = torch.stack(outputs.hidden_states, dim=0)[:, 0]        # (13, T, 768)
    emb = hs.mean(dim=1).cpu().float().numpy()                 # (13, 768)
    return emb


def process_source(key: str, processor, model, device: str, ckpt_every: int = 100):
    cfg = SOURCES[key]
    df = pd.read_csv(cfg['meta'])
    paths = df[cfg['path_col']].tolist()
    n = len(paths)

    # 체크포인트 재개
    ckpt_dir = cfg['ckpt']; ckpt_dir.mkdir(exist_ok=True)
    emb_ckpt = ckpt_dir / 'emb.npy'
    done = 0
    embeddings = []
    if emb_ckpt.exists():
        prev = np.load(emb_ckpt)
        embeddings = [prev[i] for i in range(len(prev))]
        done = len(embeddings)
        print(f"  [{key}] 체크포인트 재개: {done}/{n} 완료", flush=True)

    t0 = time.time()
    for i in range(done, n):
        p = paths[i] if cfg['abs'] else str(ROOT / paths[i])
        embeddings.append(extract_one(p, processor, model, device))
        if (i + 1) % ckpt_every == 0 or (i + 1) == n:
            np.save(emb_ckpt, np.array(embeddings, dtype=np.float32))
            el = time.time() - t0
            rate = (i + 1 - done) / el if el > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{key}] {i+1:>4}/{n}  {rate:.1f} file/s  ETA {eta/60:.1f}min", flush=True)

    arr = np.array(embeddings, dtype=np.float32)   # (n, 13, 768)
    np.save(cfg['out'], arr)
    print(f"  [{key}] 저장: {cfg['out'].name}  shape={arr.shape}", flush=True)
    return arr


def verify(key: str, atol: float = 1e-3):
    """마지막 층(12) mean-pool이 기존 임베딩과 일치하는지 확인."""
    cfg = SOURCES[key]
    if not cfg['out'].exists() or not cfg['ref'].exists():
        print(f"  [{key}] 검증 스킵 (파일 없음)"); return
    new = np.load(cfg['out'])          # (n, 13, 768)
    ref = np.load(cfg['ref']).astype(np.float32)  # (n, 768)
    last = new[:, -1, :]               # 마지막 층
    if new.shape[0] != ref.shape[0]:
        print(f"  [{key}] ❌ 행 수 불일치 {new.shape[0]} vs {ref.shape[0]}"); return
    max_abs = np.abs(last - ref).max()
    mean_abs = np.abs(last - ref).mean()
    ok = max_abs < atol
    print(f"  [{key}] 마지막 층 vs 기존: max|Δ|={max_abs:.2e} mean|Δ|={mean_abs:.2e} "
          f"→ {'✅ 정렬 일치' if ok else '⚠️ 불일치(추출 설정 확인)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['orch', 'phil'], default=None)
    ap.add_argument('--verify', action='store_true', help='추출 없이 검증만')
    args = ap.parse_args()

    keys = [args.only] if args.only else ['orch', 'phil']

    if args.verify:
        for k in keys:
            verify(k)
        return

    device = get_device()
    processor, model = load_mert(device)
    for k in keys:
        process_source(k, processor, model, device)
    print("\n=== 검증 ===")
    for k in keys:
        verify(k)


if __name__ == '__main__':
    main()
