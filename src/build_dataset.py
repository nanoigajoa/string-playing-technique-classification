"""
특징 추출 실행 스크립트.
OrchideaSOL 현악기 오디오 전체를 읽어 특징 벡터를 추출하고 data/features.csv로 저장한다.

실행:
    python src/build_dataset.py
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import extract_features, make_feature_names, parse_recording_id, FEATURE_NAMES

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = PROJECT_ROOT / '_OrchideaSOL2020_release' / 'OrchideaSOL2020' / 'Strings'
OUTPUT_PATH  = PROJECT_ROOT / 'data' / 'features.csv'

# ── 클래스 / 악기 설정 ───────────────────────────────────────────────────────
CLASSES = [
    'ordinario',
    'pizzicato_secco',
    'tremolo',
    'sul_ponticello',
    'col_legno_battuto',
]
INSTRUMENTS = ['Violin', 'Viola', 'Violoncello', 'Contrabass']
LABEL2ID    = {c: i for i, c in enumerate(CLASSES)}

# 클래스당 최대 샘플 수 (None = 전체 사용)
MAX_SAMPLES_PER_CLASS = 200


def scan_files() -> pd.DataFrame:
    """DATA_ROOT를 순회하며 대상 파일 목록을 DataFrame으로 반환"""
    rows = []
    for instrument in INSTRUMENTS:
        for technique in CLASSES:
            folder = DATA_ROOT / instrument / technique
            if not folder.exists():
                print(f'  [없음] {instrument}/{technique}')
                continue
            for f in sorted(folder.glob('*.wav')):
                rows.append({
                    'filepath':     str(f),
                    'label':        technique,
                    'label_id':     LABEL2ID[technique],
                    'instrument':   instrument,
                    'recording_id': parse_recording_id(f.name),
                })
    return pd.DataFrame(rows)


def downsample(df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    return (
        df.groupby('label', group_keys=False)
          .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
          .reset_index(drop=True)
    )


def main():
    print(f'데이터 경로: {DATA_ROOT}')
    assert DATA_ROOT.exists(), f'데이터 폴더가 없습니다: {DATA_ROOT}'

    # 파일 스캔
    df_all = scan_files()
    print(f'\n전체 파일 수: {len(df_all)}')
    print(df_all.groupby('label')['filepath'].count().to_string())

    # 다운샘플링
    df = downsample(df_all, MAX_SAMPLES_PER_CLASS) if MAX_SAMPLES_PER_CLASS else df_all
    print(f'\n다운샘플 후: {len(df)} 파일')

    # 특징 추출
    records, errors = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='특징 추출'):
        try:
            vec    = extract_features(row['filepath'])
            record = dict(zip(FEATURE_NAMES, vec))
            record.update({
                'label':        row['label'],
                'label_id':     row['label_id'],
                'instrument':   row['instrument'],
                'recording_id': row['recording_id'],
            })
            records.append(record)
        except Exception as e:
            errors.append((row['filepath'], str(e)))

    print(f'\n성공: {len(records)} / 실패: {len(errors)}')
    for path, msg in errors[:5]:
        print(f'  오류: {Path(path).name} — {msg}')

    # 저장
    meta_cols = ['label', 'label_id', 'instrument', 'recording_id']
    df_feat   = pd.DataFrame(records)[meta_cols + FEATURE_NAMES]

    # 기본 검증
    nan_count = df_feat[FEATURE_NAMES].isna().sum().sum()
    if nan_count > 0:
        print(f'경고: NaN {nan_count}개 발견')

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df_feat.to_csv(OUTPUT_PATH, index=False)
    print(f'\n저장 완료: {OUTPUT_PATH}')
    print(f'shape: {df_feat.shape}')


if __name__ == '__main__':
    main()
