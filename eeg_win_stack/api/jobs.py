"""Importable, path-explicit orchestration for the EEG win-stack API.

These functions are the real work behind every entrypoint — the LocalBackend, the
CLI, and the FastAPI service all funnel here. Unlike ``pipeline/train.py`` (a
procedural DVC stage with hardcoded paths and no return value), they take a
resolved config dict and explicit I/O paths and *return a result object*, so they
can be driven from code, a request handler, or a remote job alike.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from braindecode.datautil import load_concat_dataset

from eeg_win_stack.api.artifacts import ModelArtifact
from eeg_win_stack.models import ModelFactory
from eeg_win_stack.tools.dataset_splitting import DatasetSplitter
from eeg_win_stack.training.trainer import Trainer, TrainingConfig


@dataclass
class TrainResult:
    """Outcome of :func:`run_training`.

    Attributes
    ----------
    model_id : str
        Identifier of the saved artifact.
    model_path : pathlib.Path
        Path to the saved ``.pt`` weights.
    manifest_path : pathlib.Path
        Path to the saved ``.json`` manifest.
    artifact : ModelArtifact
        The full artifact, ready to reload for evaluation or inference.
    """

    model_id: str
    model_path: Path
    manifest_path: Path
    artifact: ModelArtifact


def _model_build_kwargs(
    model_cfg: dict, *, n_classes: int, n_channels: int, window_len_samples: int
) -> dict:
    """Assemble the kwargs for :meth:`ModelFactory.create` from the ``[model]`` config.

    Only the subsection matching the model name is applied (e.g. ``[model.deep4]``
    for ``name = "deep4"``). This differs deliberately from the old pipeline, which
    spread *every* ``[model.*]`` subsection into one ``create`` call — with the
    default config that raises ``TypeError`` on keys shared between ``deep4`` and
    ``shallow``, and otherwise lets one model's hyperparameters leak into another.

    Parameters
    ----------
    model_cfg : dict
        The ``[model]`` config section, including per-model subsections.
    n_classes : int
        Number of output classes.
    n_channels : int
        Number of input channels, taken from the windowed data.
    window_len_samples : int
        Input window length in samples, taken from the windowed data.

    Returns
    -------
    dict
        The exact kwargs to pass to ``ModelFactory.create(name, **kwargs)``; also
        stored verbatim in the artifact manifest for deterministic reloading.
    """
    build_kwargs = {
        "n_channels": n_channels,
        "n_classes": n_classes,
        "input_window_samples": window_len_samples,
        "drop_prob": model_cfg["dropout"],
        "final_conv_length": model_cfg["final_conv_length"],
    }
    build_kwargs.update(model_cfg.get(model_cfg["name"], {}))
    return build_kwargs


def _training_config(training_cfg: dict) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from the ``[training]`` config section."""
    return TrainingConfig(
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        batch_size=training_cfg["batch_size"],
        n_epochs=training_cfg["n_epochs"],
        early_stopping=training_cfg["earlystopping"],
        es_threshold=training_cfg["es_threshold"],
        es_patience=training_cfg["es_patience"],
        test_on_eval=training_cfg["test_on_eval"],
        checkpoint_dir=training_cfg["checkpoint_dir"],
    )


def run_training(config: dict, *, windows_path, output_dir, model_id: str | None = None) -> TrainResult:
    """Train a model from windowed data and save it as a :class:`ModelArtifact`.

    The library form of ``pipeline/train.py``: load windows, split, build the
    model, fit, and persist a weights + manifest pair.

    Parameters
    ----------
    config : dict
        Resolved configuration (as returned by :func:`eeg_win_stack.config.load`),
        with at least the ``training``, ``model``, ``split``, and ``run`` sections.
    windows_path : str or pathlib.Path
        Directory of saved windowed data to load via ``load_concat_dataset``.
    output_dir : str or pathlib.Path
        Directory the artifact pair is written into.
    model_id : str, optional
        Identifier for the artifact. Defaults to ``"<name>_<timestamp>"``.

    Returns
    -------
    TrainResult
        The saved model's id, file paths, and artifact handle.
    """
    training_cfg = config["training"]
    model_cfg = config["model"]
    split_cfg = config["split"]
    run_cfg = config["run"]

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(run_cfg["n_jobs"])

    windows_ds = load_concat_dataset(
        path=str(windows_path),
        preload=False,
        target_name="pathological",
        n_jobs=1,
    )

    splitter = DatasetSplitter(
        windows_ds,
        split_cfg["train_size"],
        split_cfg["valid_size"],
        split_cfg["test_size"],
        run_cfg["random_state"],
        shuffle=split_cfg["shuffle"],
        remove_attribute=None,
    )
    train_set, valid_set, _ = splitter.split_data(split_cfg["split_way"])

    n_channels = windows_ds[0][0].shape[0]
    window_len_samples = windows_ds[0][0].shape[1]

    build_kwargs = _model_build_kwargs(
        model_cfg,
        n_classes=training_cfg["n_classes"],
        n_channels=n_channels,
        window_len_samples=window_len_samples,
    )
    model = ModelFactory.create(model_cfg["name"], **build_kwargs)

    training_config = _training_config(training_cfg)
    eeg_classifier = Trainer(training_config).fit(model, train_set, valid_set)

    if model_id is None:
        model_id = model_cfg["name"] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

    artifact = ModelArtifact.save(
        eeg_classifier,
        model_id=model_id,
        model_name=model_cfg["name"],
        build_kwargs=build_kwargs,
        output_dir=output_dir,
        training_config=asdict(training_config),
    )

    return TrainResult(
        model_id=artifact.model_id,
        model_path=artifact.model_path,
        manifest_path=artifact.manifest_path,
        artifact=artifact,
    )
