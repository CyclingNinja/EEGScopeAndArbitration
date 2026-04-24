# Windows Environment Setup Guide

This guide explains how to replicate the Python 3.9 environment for this project on Windows using **uv**, a fast Python package manager.

---

## 1. Install uv

Open **PowerShell** (search for it in the Start menu) and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell after installation so the `uv` command is available.

Verify it worked:

```powershell
uv --version
```

---

## 2. Clone the repository

If you do not already have the project folder, clone it (requires [Git for Windows](https://git-scm.com/download/win)):

```powershell
git clone <repository-url>
cd EEGScopeAndArbitration
```

---

## 3. Install Python 3.9

uv can download and manage Python versions for you:

```powershell
uv python install 3.9
```

---

## 4. Create the virtual environment

From inside the project folder, run:

```powershell
uv venv --python 3.9 .venv
```

This creates a `.venv` folder in the project directory.

---

## 5. Install project dependencies

### Option A — CPU only (sufficient for reviewing results, not for training)

```powershell
uv pip install --python .venv\Scripts\python.exe -e .
```

### Option B — GPU / CUDA 11.7 (required for training on an NVIDIA GPU)

```powershell
uv pip install --python .venv\Scripts\python.exe -e . --extra-index-url https://download.pytorch.org/whl/cu117
```

> **Which option?** If you are only inspecting results or running short evaluations, Option A is fine. Training on the full TUAB dataset requires a GPU (Option B).

---

## 6. Activate the environment

Each time you open a new terminal to work on the project:

```powershell
.venv\Scripts\activate
```

Your prompt will change to show `(.venv)` when it is active.

---

## 7. Running the project

Follow the steps in [README.md](README.md):

1. Download the TUAB dataset and place it in the `data/` folder.
2. Copy the config templates and fill in your parameters:
   ```powershell
   copy batch_test_hyperparameters.default.py batch_test_hyperparameters.py
   copy train_and_eval_config.default.py train_and_eval_config.py
   ```
3. Edit `batch_test_hyperparameters.py` and `train_and_eval_config.py` as needed.
4. Run training:
   ```powershell
   python train_and_eval.py
   ```
5. Run the second-stage decision model:
   ```powershell
   python final_decision.py
   ```

Results are written to `result.csv` and `training_detail.csv`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv` not found after install | Restart PowerShell, or add `%USERPROFILE%\.local\bin` to your PATH |
| `python` not found inside `.venv` | Make sure you ran step 4 from inside the project folder |
| CUDA errors at runtime | Check your GPU driver supports CUDA 11.7; alternatively use Option A (CPU) |
| `ExecutionPolicy` error on activation | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in PowerShell |
