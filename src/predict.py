"""
추론 파이프라인.
오디오 파일 하나를 입력받아 연주기법 예측 결과와 확신도를 반환한다.

사용법:
    python src/predict.py path/to/audio.wav
    python src/predict.py path/to/audio.wav --model cnn
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

CLASSES = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
MODELS_DIR = ROOT / 'models'


def predict_svm(audio_path: str) -> dict:
    """v1 SVM 파이프라인으로 예측 (특징 추출 → StandardScaler → SVM)"""
    import joblib
    from features import extract_features

    model_path = MODELS_DIR / 'svm_v1.joblib'
    if not model_path.exists():
        raise FileNotFoundError(
            f'{model_path} 없음. 02_modeling_v1.ipynb를 실행해 모델을 먼저 저장하세요.'
        )

    pipeline = joblib.load(model_path)
    feat = extract_features(str(audio_path)).reshape(1, -1)

    pred_idx = pipeline.predict(feat)[0]
    # SVC(probability=False) 기본값이므로 decision_function으로 확신도 근사
    if hasattr(pipeline.named_steps['svm'], 'predict_proba'):
        proba = pipeline.predict_proba(feat)[0]
    else:
        scores = pipeline.decision_function(feat)[0]
        # softmax로 확신도 근사
        exp_s = np.exp(scores - scores.max())
        proba = exp_s / exp_s.sum()

    return {
        'model': 'SVM (v1)',
        'predicted': CLASSES[pred_idx],
        'confidence': float(proba[pred_idx]),
        'probabilities': {cls: float(p) for cls, p in zip(CLASSES, proba)},
    }


def predict_cnn(audio_path: str) -> dict:
    """v2 CNN으로 예측 (멜 스펙트로그램 → SmallCNN)"""
    import torch
    from dataset import audio_to_mel
    from model import SmallCNN

    model_path = MODELS_DIR / 'cnn_v2.pt'
    if not model_path.exists():
        raise FileNotFoundError(
            f'{model_path} 없음. 03_modeling_v2.ipynb를 실행해 모델을 먼저 저장하세요.'
        )

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    model = SmallCNN(n_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    mel = audio_to_mel(str(audio_path))
    x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)[0]
        proba = torch.softmax(logits, dim=0).cpu().numpy()

    pred_idx = int(proba.argmax())
    return {
        'model': 'CNN (v2)',
        'predicted': CLASSES[pred_idx],
        'confidence': float(proba[pred_idx]),
        'probabilities': {cls: float(p) for cls, p in zip(CLASSES, proba)},
    }


def predict(audio_path: str, model: str = 'svm') -> dict:
    """
    오디오 파일 경로를 받아 연주기법을 예측한다.

    Args:
        audio_path: WAV 파일 경로
        model: 'svm' (v1, 기본값) 또는 'cnn' (v2)

    Returns:
        {
            'model': str,
            'predicted': str,       # 예측된 연주기법
            'confidence': float,    # 예측 확신도 (0~1)
            'probabilities': dict,  # 클래스별 확률
        }
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f'오디오 파일을 찾을 수 없습니다: {audio_path}')

    if model == 'svm':
        return predict_svm(audio_path)
    elif model == 'cnn':
        return predict_cnn(audio_path)
    else:
        raise ValueError(f"model은 'svm' 또는 'cnn'이어야 합니다. 입력값: {model!r}")


def main():
    parser = argparse.ArgumentParser(description='현악기 연주기법 분류')
    parser.add_argument('audio', help='입력 오디오 파일 경로 (.wav)')
    parser.add_argument(
        '--model', choices=['svm', 'cnn'], default='svm',
        help='사용할 모델 (기본값: svm)'
    )
    args = parser.parse_args()

    result = predict(args.audio, model=args.model)

    print(f'\n[{result["model"]}] {Path(args.audio).name}')
    print(f'  예측 기법:  {result["predicted"]}')
    print(f'  확신도:     {result["confidence"]:.1%}')
    print('\n  클래스별 확률:')
    for cls, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = '█' * int(p * 30)
        print(f'    {cls:<22} {p:.3f}  {bar}')


if __name__ == '__main__':
    main()
