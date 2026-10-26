# Repository Guidelines

## Project Structure & Module Organization

This repository studies five bowed-string playing techniques using audio ML. Keep reusable Python code in `src/`: `features.py` implements the 245-feature v1 pipeline; `dataset.py` and `model.py` support the mel-spectrogram CNN; `phase1_*.py` through `phase4_*.py` are research experiments. Use `notebooks/` for exploratory work and baseline training, `results/` for generated reports, plots, and tables, and `docs/` or `연구과정/` for methodology and manuscript material. Trained artifacts belong in `models/`. Do not commit raw OrchideaSOL audio or regenerated embedding caches.

## Setup, Development, and Reproduction

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_modeling_v1.ipynb
python src/predict.py path/to/audio.wav --model cnn
```

Use `python src/build_dataset.py` to rebuild the feature matrix when the local OrchideaSOL release is available. Run experiment scripts explicitly, for example `python src/phase1_mert_tuning_audit.py` or `python src/phase3_shap.py`; inspect their output under `results/`. Prefer MPS where supported and retain CPU fallbacks. Do not introduce CUDA-only dependencies such as `bitsandbytes` or paid API calls.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and focused modules. Name phase scripts as `phase<N>_<purpose>.py` and keep generated artifacts grouped by phase, e.g. `results/phase3/report.md`. Add concise docstrings or comments where feature shapes, splits, or audio preprocessing are non-obvious. No formatter or linter is configured; preserve surrounding style and avoid unrelated reformatting.

## Testing & Evaluation

There is no dedicated automated test suite. For changes, run the smallest relevant script or notebook section and verify saved metrics and plots. Evaluation must use instrument-level groups—`GroupKFold` / LOSO—not clip-level random splits. Report macro F1, per-class behavior, and confusion matrices; preserve valid negative controls such as chroma ablations.

## Commits & Pull Requests

Recent commits use short, imperative summaries, often bilingual, such as `Update v2 results...` or `README 개편: ...`. Keep each commit scoped to one change. PRs should state the research question, scripts/notebooks run, dataset or split assumptions, and affected result files. Include before/after metrics or plot screenshots for model or documentation claims, and never present test-selected checkpoints as held-out results.
