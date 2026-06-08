"""DVC evaluate stage: load model, run Evaluator, log to MLflow, write metrics.json."""

import json
import mne
from pathlib import Path

import mlflow
import torch
from braindecode.datautil import load_concat_dataset

from eeg_win_stack.config import load
from eeg_win_stack.evaluation.evaluator import Evaluator
from eeg_win_stack.models import ModelFactory
from eeg_win_stack.tools.dataset_splitting import DatasetSplitter
from eeg_win_stack.training.trainer import Trainer, TrainingConfig


def main():
    cfg = load()
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    split_cfg = cfg["split"]
    run_cfg = cfg["run"]

    mne.set_log_level(run_cfg["mne_log_level"])

    windows_ds = load_concat_dataset(
        path="data/saved_windows",
        preload=False,
        target_name="pathological",
        n_jobs=1,
    )

    data_choice = DatasetSplitter(
        windows_ds,
        split_cfg["train_size"],
        split_cfg["valid_size"],
        split_cfg["test_size"],
        run_cfg["random_state"],
        shuffle=split_cfg["shuffle"],
        remove_attribute=None,
    )
    _, _, test_set = data_choice.split_data(split_cfg["split_way"])

    n_channels = windows_ds[0][0].shape[0]
    window_len_samples = windows_ds[0][0].shape[1]

    model = ModelFactory.create(
        model_cfg["name"],
        n_channels=n_channels,
        n_classes=training_cfg["n_classes"],
        input_window_samples=window_len_samples,
        drop_prob=model_cfg["dropout"],
        final_conv_length=model_cfg["final_conv_length"],
        **model_cfg.get("deep4", {}),
        **model_cfg.get("tcn", {}),
        **model_cfg.get("shallow", {}),
        **model_cfg.get("vit", {}),
    )

    params_path = sorted(Path("data/saved_models").glob("*.pt"))[-1]
    training_config = TrainingConfig(
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        batch_size=training_cfg["batch_size"],
        n_epochs=training_cfg["n_epochs"],
    )
    eeg_classifier = Trainer(training_config).load(model, params_path)

    result = Evaluator().evaluate(eeg_classifier, test_set)

    metrics = {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "mcc": result.mcc,
    }

    Path("metrics.json").write_text(json.dumps(metrics, indent=2))

    mlflow.set_tracking_uri("mlruns")
    with mlflow.start_run():
        mlflow.log_params({
            "model": model_cfg["name"],
            "learning_rate": training_cfg["learning_rate"],
            "weight_decay": training_cfg["weight_decay"],
            "batch_size": training_cfg["batch_size"],
            "n_epochs": training_cfg["n_epochs"],
            "split_way": split_cfg["split_way"],
        })
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
