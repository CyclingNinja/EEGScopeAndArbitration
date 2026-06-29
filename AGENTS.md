# AI Agent Instructions for EEGScopeAndArbitration

## Project Overview

**EEG Scope & Arbitration** is a window-level EEG abnormality classification system for detecting abnormal patterns in electroencephalography recordings. The project loads raw EEG data (TUAB/TUEG datasets), preprocesses and windows recordings into fixed 60-second segments, trains deep learning models, and evaluates them for binary classification.

The core functionality lives in the **`eeg_win_stack`** package as a production-ready API layer, separate from the experimental DVC pipeline for parameter sweeps.

## Quick Start for Agents

### Critical Architecture Principle
**Three-layer design**: Orchestration functions → Backend abstraction → Execution environment
- **Production API** (`eeg_win_stack/api/`): Use `run_training()` and `run_evaluation()` for reproducible runs
- **DVC Pipeline** (`eeg_win_stack/pipeline/`): Experimentation and parameter sweeps only
- **Backends** (`eeg_win_stack/api/backends/`): Abstract where jobs run (local now, cloud later)

### Key Modules at a Glance
```
eeg_win_stack/
  api/          → Production interface: jobs.py, artifacts.py, backends/
  io/           → Data loading: RawEEGLoader, DatasetBuilder (3-tier load priority)
  models/       → ModelFactory + registered architectures (Deep4, EEGNet, TCN, etc.)
  training/     → Trainer + TrainingConfig (braindecode wrapper)
  evaluation/   → Evaluator (window-level metrics)
  tools/        → Utilities: splitting, metrics, paths, plotting
  config/       → params.toml + loader
  pipeline/     → DVC stages (experimentation only)
  tests/        → Comprehensive test suite with fixtures
```

### Build & Test Commands

**Run all tests with coverage:**
```bash
tox
```

**Run tests directly (faster for development):**
```bash
pytest eeg_win_stack --cov
```

**Run linting only:**
```bash
tox -e lint
```

**Run specific test file:**
```bash
pytest eeg_win_stack/tests/training/test_trainer.py -vvv
```

**Run full experimental pipeline (slow):**
```bash
dvc repro
```

**Legacy monolithic script:**
```bash
python train_and_eval.py
```

## Configuration & Conventions

### Configuration System
- **Single source of truth**: `eeg_win_stack/config/params.toml` (TOML format)
- **Loading**: `from eeg_win_stack.config import load; config = load()`
- **Factory pattern**: `ModelFactory` intelligently filters model kwargs based on architecture
- **Key sections**:
  - `[run]`: Repetitions, random state, parallelism
  - `[data]`: Dataset selection (TUAB/TUEG), paths, caching
  - `[preprocessing]`: Filtering, standardization, sampling rate (100 Hz fixed)
  - `[windowing]`: Window length (60s), 19-channel 10-20 EEG system
  - `[split]`: Train/val/test ratios (0.8/0.1/0.1 default)
  - `[training]`: Learning rate, batch size, epochs, early stopping
  - `[model]`: Model name + per-model hyperparameters

### Model Artifacts & Portability
Models are saved as **checkpoint + manifest pair**:
- `<model_id>.pt` — weights only (skorch/braindecode format)
- `<model_id>.json` — self-describing manifest (format version, build kwargs, training config)

**Why**: Manifest makes models portable and deterministic to reload without original training config or dataset.

### Data Loading (3-Tier Priority)
`DatasetBuilder` attempts to load in this order:
1. **Saved windows** — Pre-windowed EEG (fastest, cached after first preprocessing)
2. **Saved preprocessed** — Preprocessed but not windowed
3. **Raw sources** — TUAB/TUEG EDF files (slowest, requires full pipeline)

Control via `params.toml` flags: `load_saved_windows`, `load_saved_recordings`, `save_windows`

### Fixed Preprocessing Chain
- **Sampling**: Always 100 Hz (resampled in MNE load)
- **19 channels**: Standard 10-20 EEG system (FP, F, C, P, O, T, Z regions)
- **Standardization**: Robust scaling with `factor_new=1e-3` and `init_block_size=1000`
- **Clipping**: Outlier clipping at ±800 µV
- **Windowing**: 60-second windows (6000 samples), 300s initial skip, 60s from end
- **Optional bandpass**: 4–38 Hz (disabled by default in params.toml)

## Common Development Patterns

### Training Pipeline
```python
from eeg_win_stack.config import load
from eeg_win_stack.api import run_training

config = load()
result = run_training(
    config,
    windows_path="data/saved_windows",
    output_dir="data/saved_models",
)
print(result.model_id, result.model_path, result.manifest_path)
```

### Evaluation Pipeline
```python
from eeg_win_stack.api import run_evaluation

result = run_evaluation(
    config,
    model_path="data/saved_models/deep4_2026-06-10_14-38-14.pt",
    model_manifest_path="data/saved_models/deep4_2026-06-10_14-38-14.json",
    windows_path="data/saved_windows",
)
print(result.accuracy, result.mcc, result.confusion_matrix)
```

### Backend Abstraction (for jobs/CLI)
```python
from eeg_win_stack.api.backends import get_backend, Job, JobKind
from eeg_win_stack.api.jobs import run_training

backend = get_backend("local")  # or "azureml", "slurm" later
job = Job(kind=JobKind.TRAINING, config=config, ...)
handle = backend.submit(job)
status = backend.status(handle)
result = backend.result(handle)
```

## Important Notes for Agents

### Python Version
- **Locked to 3.9** — torch/braindecode wheels only available for 3.9 currently
- Python 3.10+ migration is deferred; do not upgrade without consulting
- `tox.ini` enforces py39 only

### Testing Requirements
- **Test framework**: pytest with pytest-cov
- **Fixtures**: Comprehensive conftest.py in `eeg_win_stack/tests/io/` for dataset mocking
- **Expected coverage**: Run tests before commits
- **Lint standard**: ruff (check + format)

### Dataset Handling
- **TUAB path**: Configured in `params.toml` (currently `/home/sam/Code/.../data/TUAB` — **will fail if not present**)
- **Alternative TUEG**: Available but disabled by default
- **Large files**: First run preprocesses entire dataset and caches to `data/saved_windows/` (can take hours)
- **Preload flag**: If `preload=true`, entire dataset loads to RAM (requires ~8GB for defaults)

### Model Selection
- **Default model**: Deep4 (small, fast for development)
- **Available architectures**: Deep4, EEGNet (v1/v4), EEGResNet, Shallow, TCN, SleepStager variants, TIDNet, USleep
- **Model factory**: Intelligently filters kwargs; always use `ModelFactory.create(model_name, **build_kwargs)`

### Logging & Output
- **CSV logging**: `data/target/result.csv` — results logged row-by-row (via legacy `train_and_eval.py`)
- **Results dataframe**: `result.csv` contains metrics per run
- **Metrics**: Window-level accuracy, precision, recall, MCC, confusion matrix
- **Plotting**: Optional via `plot_result=true` in params (requires matplotlib)

## Dependencies & Environment

### Runtime Dependencies
- **Core ML**: torch (2.2–2.3), braindecode (0.6), skorch (0.15), scikit-learn (1.4)
- **EEG I/O**: mne (1.6), einops (0.7)
- **Data**: numpy (1.26), pandas (2.2)
- **Optimization**: bayesian-optimization (1.4.3), xgboost (2.0)
- **Logging**: mlflow (2.0)
- **Config**: tomli (2.0)

### Development Dependencies
- **Testing**: pytest (8.4.2+), pytest-cov
- **Linting**: ruff (0.15.14+)
- **Environment**: tox (4.22+), tox-uv (1.13+)

### Installation
```bash
# With uv (recommended, fast)
uv sync

# Or pip
pip install -e ".[dev]"
```

## Pitfalls & Anti-Patterns

1. **Don't hardcode data paths** — Always read from `params.toml`
2. **Don't call DVC pipeline stages directly** — Use `eeg_win_stack.api` jobs instead (production API is the contract)
3. **Don't modify params.toml in tests** — Create test configs in fixtures
4. **Don't use models outside ModelFactory** — Factory applies intelligent kwarg filtering
5. **Don't assume TUAB/TUEG paths exist** — Verify in CI; tests use mocked datasets
6. **Don't reload models without manifest** — Always use manifest.json alongside .pt file
7. **Don't forget early stopping check** — Validation curve should flatten; if not, increase `es_patience` in params
8. **Don't skip windowing steps** — Preprocessing order matters (filter → standardize → window)

## Where to Start

**For bug fixes**: Check [eeg_win_stack/tests/](eeg_win_stack/tests/) for test patterns; run `pytest eeg_win_stack -vvv` to see failures.

**For new features**: 
- Data loading: [eeg_win_stack/io/](eeg_win_stack/io/)
- Models: [eeg_win_stack/models/](eeg_win_stack/models/) + add to ModelFactory
- Metrics: [eeg_win_stack/tools/metrics.py](eeg_win_stack/tools/metrics.py)
- Training: [eeg_win_stack/training/trainer.py](eeg_win_stack/training/trainer.py)
- API: [eeg_win_stack/api/jobs.py](eeg_win_stack/api/jobs.py)

**For configuration tuning**: Edit [eeg_win_stack/config/params.toml](eeg_win_stack/config/params.toml) and re-run `python train_and_eval.py` or `dvc repro`.

**For documentation**: See [README.md](README.md) for full API usage; [docs/](docs/) for DVC and Azure setup.
