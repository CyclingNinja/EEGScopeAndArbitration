# DVC Workflow Guide

An end-to-end, task-oriented guide to running the `eeg_win_stack` pipeline with
DVC and Azure Blob Storage: how to install and configure it once, and what to do
before, during, and after each run.

This guide stitches the workflow together. For the deep dives it links out to:

- **[dvc-pipeline.md](dvc-pipeline.md)** — the stage/param reference and experiment/sweep mechanics.
- **[azure-blob-setup.md](azure-blob-setup.md)** — creating the Azure storage account, containers, and keys from the portal.

> **Scope.** The DVC pipeline exists for **experimentation** — reproducible runs,
> parameter sweeps, and metric tracking. It is separate from the `eeg_win_stack/api/`
> layer (see the [README](../README.md)), which is the production interface.

---

## 1. The pipeline at a glance

Three stages, defined declaratively in [`dvc.yaml`](../dvc.yaml) and parameterised
entirely from [`eeg_win_stack/config/params.toml`](../eeg_win_stack/config/params.toml):

```
preprocess ──▶ train ──▶ evaluate
```

| Stage | Command | Produces |
| --- | --- | --- |
| `preprocess` | `python -m eeg_win_stack.pipeline.preprocess` | `target/saved_windows/` (DVC-cached) |
| `train` | `python -m eeg_win_stack.pipeline.train` | `target/saved_models/` (DVC-cached) |
| `evaluate` | `python -m eeg_win_stack.pipeline.evaluate` | `metrics.json` (git-tracked) + an MLflow run |

DVC runs stages in order and **skips any stage whose code, deps, and params are
unchanged** since the last run. All large outputs live under `target/` (git-ignored;
DVC manages their contents and syncs them to Azure).

---

## 2. One-time setup

### 2.1 Python environment

This branch pins **braindecode 0.6.x on Python 3.9**. Use the project's `uv`/`tox`
environment so you get a compatible interpreter and dependency set — a mismatched
braindecode will fail to import the models.

```bash
uv sync --all-groups          # create/refresh .venv from pyproject
# DVC ships in the `dev` dependency group (dvc + dvc-azure), so it lands in .venv
```

If `dvc` is not on your `PATH`, call it via the venv (`.venv/bin/dvc`) or run inside
`tox`. All commands below assume `dvc` resolves to the project environment.

### 2.2 Initialise DVC (fresh clone only)

The repo is already DVC-initialised (`.dvc/` is committed), so a normal clone needs
**no** `dvc init`. You only run it when starting DVC in a repo that has none.

### 2.3 Connect the Azure remote credentials

The remotes themselves are already committed in [`.dvc/config`](../.dvc/config):

```ini
[core]
    remote = azure                                   # default remote
['remote "medshed"']
    url = azure://medshedeegwinstack/dvc-cache
['remote "azure"']
    url = azure://dvc-cache/eeg-pipeline
```

What you must supply on each new machine is the **credential**, which is never
committed. Get the connection string from the storage account (see
[azure-blob-setup.md §1.4](azure-blob-setup.md)), then either:

```bash
# Option A — store it against the remote (writes .dvc/config.local, git-ignored)
dvc remote modify --local azure connection_string \
  "DefaultEndpointsProtocol=https;AccountName=<acct>;AccountKey=<KEY>;EndpointSuffix=core.windows.net"

# Option B — export it (DVC and MLflow's Azure backend both read this)
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=<acct>;AccountKey=<KEY>;EndpointSuffix=core.windows.net"
```

Put Option B in your shell profile (`~/.zshrc`) if you want it to persist. **Never
commit the connection string.** `.dvc/config.local` is git-ignored by DVC.

### 2.4 Pull cached data

```bash
dvc pull                              # fetch all cached outputs from the default remote
dvc pull target/saved_windows         # or just the windowed dataset
```

Pulling the windows lets you **skip `preprocess`** entirely — handy on a machine
that doesn't have the raw TUAB EDFs (`tuab_path` in `params.toml` points at a local
drive such as `E:/TUAB`). See [§5](#5-working-on-another-machine).

### 2.5 Raw data (only if you will run `preprocess`)

`preprocess` needs the TUAB/TUEG datasets on disk at the paths under `[data]` in
`params.toml` (`tuab_path`, `tueg_path`). If you only train/evaluate from pulled
windows, you can skip this.

---

## 3. Before a run (pre-flight)

1. **Sync inputs.** `git pull` for code/params, then `dvc pull` for cached data.
2. **Decide how to change parameters** (see §4). Prefer `-S` overrides for anything
   you want to compare later; edit `params.toml` directly for a durable change.
3. **Check config ↔ cached-window consistency.** The `train`/`evaluate` stages fail
   fast if the loaded windows don't match `sampling_freq × window_len_s` in the
   config (a stale cache built under a different config). If you change windowing or
   sampling params, re-run `preprocess` (or pull matching windows) before training —
   otherwise you'll get a clear `ValueError` naming the expected vs actual length.
4. **Know which model runs.** `[model].name` selects the architecture from the
   registry (e.g. `"tcn"`). If a model has a `[model.<name>]` sub-table, its keys are
   passed as constructor kwargs.

---

## 4. Making a run

### Baseline (`dvc repro`)

Runs the pipeline with the current `params.toml`, rebuilding only changed stages:

```bash
dvc repro                 # full pipeline
dvc repro train           # up to a named stage (and its deps)
dvc repro -f              # force a full rerun, ignoring the cache
```

The first run writes/updates `dvc.lock`, pinning resolved deps, params, and output
hashes. **Commit `dvc.lock`** so the run is reproducible.

### Tracked experiment (`dvc exp run`, recommended)

Captures the exact params that produced each metric set without a commit per attempt.
Override any parameter with `-S 'path:section.key=value'`:

```bash
dvc exp run \
  -S 'eeg_win_stack/config/params.toml:training.learning_rate=0.0005' \
  -S 'eeg_win_stack/config/params.toml:training.n_epochs=50'
```

Promote a good one with `dvc exp apply <exp>`; discard others with `dvc exp remove`.

### Sweeps

Queue a grid and run it together — see [`examples/sweep.sh`](../examples/sweep.sh)
and [dvc-pipeline.md §Sweeps](dvc-pipeline.md). In short:

```bash
PARAMS="eeg_win_stack/config/params.toml"
for lr in 0.001 0.0005 0.0001; do
  dvc exp run --queue -S "${PARAMS}:training.learning_rate=${lr}"
done
dvc queue run --jobs 1        # raise --jobs if you have multiple GPUs
```

---

## 5. After a run (post-flight)

### 5.1 Inspect results

```bash
dvc metrics show                       # metrics for the current workspace
dvc exp show                           # table of all experiments: params + metrics
dvc metrics diff                       # compare against the last commit
```

`metrics.json` (repo root) holds the test-set `accuracy`, `precision`, `recall`, and
`mcc`. It is `cache: false`, so it is **git-tracked** and travels with your commits.

### 5.2 MLflow

The `evaluate` stage also opens an MLflow run (tracking URI `mlruns/`, experiment
`eeg_win_stack`) logging the key params and the four metrics. Artifacts (model
checkpoints) are written to the Azure artifact root configured in `params.toml`
(`[run].azure_artifact_root`). Browse locally:

```bash
mlflow ui                              # serves the dashboard from ./mlruns
```

`mlruns/` is git-ignored — it's a local log; the artifacts themselves live in Azure.

### 5.3 Push and commit

```bash
dvc push                               # upload new cached outputs to the default remote (azure)
dvc push -r medshed                    # or target a specific remote

git add dvc.lock metrics.json
git commit -m "Pipeline run: <short description>"
```

**Order matters:** `dvc push` first (so the cache exists remotely), then commit the
`dvc.lock`/`metrics.json` that reference it. This keeps history reproducible for
anyone who later `dvc pull`s your commit.

---

## 6. Working on another machine

To reproduce or continue work elsewhere without the raw dataset:

```bash
git pull
export AZURE_STORAGE_CONNECTION_STRING="..."   # once per machine (§2.3)
dvc pull target/saved_windows                   # fetch the windowed dataset
dvc repro train                                 # train straight from windows — no preprocess
```

`train`/`evaluate` load `target/saved_windows` directly, so pulling the windows is
enough to train the currently-configured model. This avoids needing `tuab_path` to
exist on that machine.

---

## 7. What gets cached vs committed

| Path | Tracking | In git? |
| --- | --- | --- |
| `target/saved_windows/` | DVC cache (`cache: true`) | no (`target/` git-ignored) |
| `target/saved_models/` | DVC cache (`cache: true`) | no (`target/` git-ignored) |
| `metrics.json` | DVC metric (`cache: false`) | **yes** |
| `dvc.lock` | git | **yes** |
| `.dvc/config` | git | **yes** (remotes, no secrets) |
| `.dvc/config.local` | local only | no (connection string) |
| `mlruns/` | local only | no (git-ignored) |

---

## 8. Troubleshooting

- **`FileExistsError: ./target/saved_recordings already contains subdirectories`.**
  braindecode's `preprocess(overwrite=False)` refuses to write into a non-empty save
  dir from a prior partial run. Clear it first: `rm -rf target/saved_recordings target/saved_windows`,
  then rerun.
- **`ValueError: Loaded windows are N samples long but the config implies M`.** The
  cached windows were built under a different `sampling_freq`/`window_len_s` than the
  current config. Re-run `preprocess` or `dvc pull` windows that match `params.toml`.
- **`dvc pull` fetches nothing / auth errors.** The connection string isn't set on
  this machine (§2.3), or you're targeting a remote that doesn't hold the output —
  try `dvc pull -r medshed` or check `dvc remote list`.
- **Evaluate scored the wrong model.** `evaluate` loads `sorted(glob("*.pt"))[-1]`
  from `target/saved_models/` (latest by filename). Clear the directory before a
  fresh run if you've accumulated checkpoints.
- **A stage didn't rerun after you changed something.** DVC skips stages whose
  tracked inputs are unchanged. Force it with `dvc repro -f` (or `-f <stage>`), e.g.
  after editing code DVC doesn't track as an explicit dependency.

---

## 9. Command cheat-sheet

```bash
# Setup (new machine)
uv sync --all-groups
export AZURE_STORAGE_CONNECTION_STRING="..."
dvc pull

# Run
dvc repro                              # baseline
dvc exp run -S 'eeg_win_stack/config/params.toml:training.n_epochs=50'   # experiment

# Inspect
dvc metrics show
dvc exp show
mlflow ui

# Publish
dvc push
git add dvc.lock metrics.json && git commit -m "Pipeline run: ..."
```
