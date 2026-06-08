"""DVC train stage: split windowed data, build model, train, save params."""

import mne
import time
from pathlib import Path

import torch
from braindecode.datautil import load_concat_dataset

from eeg_win_stack.config import load
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

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(run_cfg["n_jobs"])

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

    train_set, valid_set, test_set = data_choice.split_data(split_cfg["split_way"])

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

    training_config = TrainingConfig(
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

    trainer = Trainer(training_config)
    eeg_classifier = trainer.fit(model, train_set, valid_set)

    save_path = Path("data/saved_models") / (
        model_cfg["name"] + time.strftime("%Y-%m-%d_%H-%M-%S") + "params.pt"
    )
    Trainer.save(eeg_classifier, save_path)


if __name__ == "__main__":
    main()
