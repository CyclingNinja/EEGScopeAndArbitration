"""DVC evaluate stage: load model, run Evaluator, log to MLflow, write metrics.json."""

import json
import mne
from pathlib import Path

import mlflow
from braindecode.datautil import load_concat_dataset

from eeg_win_stack.config import load
from eeg_win_stack.evaluation.evaluator import Evaluator
from eeg_win_stack.models import ModelFactory
from eeg_win_stack.pipeline.validation import validate_window_length
from eeg_win_stack.tools.dataset_splitting import DatasetSplitter
from eeg_win_stack.tools.decision_utils import save_training_detail
from eeg_win_stack.training.trainer import Trainer, TrainingConfig
from eeg_win_stack.tools.logger import Logger, get_logger

log = get_logger(__name__)


def configure_mlflow(cfg):
    mlflow.set_tracking_uri("mlruns")

    experiment_name = cfg["run"]["experiment_name"]
    use_azure_artifacts = cfg["run"].get("use_azure_artifacts", False)
    experiment_name = experiment_name if use_azure_artifacts else f"{experiment_name}_local"

    try:
        if mlflow.get_experiment_by_name(experiment_name) is None:
            if use_azure_artifacts:
                mlflow.create_experiment(
                    experiment_name,
                    artifact_location=cfg["run"]["azure_artifact_root"],
                )
            else:
                mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
    except Exception as exc:  #  noqa: BLE001 - defensive fallback for Azure/auth issues
        log.warning("MLflow experiment setup skipped: %s", exc)
        mlflow.set_experiment(experiment_name)


def main():
    cfg = load()
    Logger.from_config(cfg)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    split_cfg = cfg["split"]
    run_cfg = cfg["run"]
    output_cfg = cfg["output"]

    mne.set_log_level(run_cfg["mne_log_level"])

    windows_ds = load_concat_dataset(
        path=cfg["data"]["save_windows_path"],
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
    train_set, valid_set, test_set = data_choice.split_data(split_cfg["split_way"])

    n_channels = windows_ds[0][0].shape[0]
    window_len_samples = windows_ds[0][0].shape[1]
    validate_window_length(window_len_samples, cfg)

    model = ModelFactory.create(
        model_cfg["name"],
        n_channels=n_channels,
        n_classes=training_cfg["n_classes"],
        input_window_samples=window_len_samples,
        drop_prob=model_cfg["dropout"],
        final_conv_length=model_cfg["final_conv_length"],
        **model_cfg.get(model_cfg["name"], {}),
    )

    params_path = sorted(Path(cfg["output"]["saved_models_path"]).glob("*.pt"))[-1]
    training_config = TrainingConfig(
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        batch_size=training_cfg["batch_size"],
        n_epochs=training_cfg["n_epochs"],
    )
    eeg_classifier = Trainer(training_config).load(model, params_path)

    save_training_detail(
        eeg_classifier,
        [("train", train_set), ("valid", valid_set), ("test", test_set)],
        output_cfg.get("training_detail_path", "target/training_detail"),
    )

    result = Evaluator().evaluate(eeg_classifier, test_set)

    metrics = {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "mcc": result.mcc,
    }

    Path("metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        configure_mlflow(cfg)
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model": model_cfg["name"],
                    "learning_rate": training_cfg["learning_rate"],
                    "weight_decay": training_cfg["weight_decay"],
                    "batch_size": training_cfg["batch_size"],
                    "n_epochs": training_cfg["n_epochs"],
                    "split_way": split_cfg["split_way"],
                }
            )
            mlflow.log_metrics(metrics)
            latest_pt = max(
                Path(cfg["output"]["saved_models_path"]).glob("*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            mlflow.log_artifact(str(latest_pt), artifact_path="model")
    except Exception as exc:  #  noqa: BLE001 - defensive fallback for Azure/auth issues
        log.warning("MLflow logging skipped because of an error: %s", exc)


if __name__ == "__main__":
    main()
