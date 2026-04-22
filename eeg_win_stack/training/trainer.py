"""EEG model training."""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from braindecode import EEGClassifier
from braindecode.models import (
    Deep4Net,
    EEGNetv1,
    EEGNetv4,
    EEGResNet,
    HybridNet,
    ShallowFBCSPNet,
    SleepStagerBlanco2020,
    SleepStagerChambon2018,
    TCN,
    TIDNet,
    USleep,
)
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch.helper import predefined_split
from torch.nn.functional import elu, gelu, relu

from eeg_win_stack.tools.metrics import weight_function


def build_model(
    model_name: str,
    n_channels: int,
    n_classes: int,
    window_len_samples: int,
    sampling_freq: float,
    *,
    activation: str = "elu",
    dropout: float = 0.2,
    final_conv_length: str | int = "auto",
    **kwargs,
) -> nn.Module:
    """Construct a model by name. Model-specific hyperparameters are passed via kwargs."""
    nonlin = {"elu": elu, "relu": relu, "gelu": gelu}[activation]

    if model_name == "deep4":
        return Deep4Net(
            n_channels,
            n_classes,
            input_window_samples=window_len_samples,
            final_conv_length=final_conv_length,
            n_filters_time=kwargs.get("deep4_n_filters_time", 25),
            n_filters_spat=kwargs.get("deep4_n_filters_spat", 25),
            filter_time_length=kwargs.get("deep4_filter_time_length", 10),
            pool_time_length=kwargs.get("deep4_pool_time_length", 3),
            pool_time_stride=kwargs.get("deep4_pool_time_stride", 3),
            n_filters_2=kwargs.get("deep4_n_filters_2", 50),
            filter_length_2=kwargs.get("deep4_filter_length_2", 10),
            n_filters_3=kwargs.get("deep4_n_filters_3", 100),
            filter_length_3=kwargs.get("deep4_filter_length_3", 10),
            n_filters_4=kwargs.get("deep4_n_filters_4", 200),
            filter_length_4=kwargs.get("deep4_filter_length_4", 10),
            first_pool_mode=kwargs.get("deep4_first_pool_mode", "max"),
            later_pool_mode=kwargs.get("deep4_later_pool_mode", "max"),
            drop_prob=dropout,
            double_time_convs=False,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False,
            first_nonlin=nonlin,
            later_nonlin=nonlin,
        )
    elif model_name == "shallow_smac":
        return ShallowFBCSPNet(
            n_channels,
            n_classes,
            input_window_samples=window_len_samples,
            n_filters_time=kwargs.get("shallow_n_filters_time", 40),
            filter_time_length=kwargs.get("shallow_filter_time_length", 25),
            n_filters_spat=kwargs.get("shallow_n_filters_spat", 40),
            pool_time_length=kwargs.get("shallow_pool_time_length", 75),
            pool_time_stride=kwargs.get("shallow_pool_time_stride", 15),
            final_conv_length=final_conv_length,
            split_first_layer=kwargs.get("shallow_split_first_layer", True),
            batch_norm=kwargs.get("shallow_batch_norm", True),
            batch_norm_alpha=kwargs.get("shallow_batch_norm_alpha", 0.1),
            drop_prob=dropout,
        )
    elif model_name == "eegnetv4":
        return EEGNetv4(
            n_channels,
            n_classes,
            input_window_samples=window_len_samples,
            final_conv_length=final_conv_length,
            pool_mode="mean",
            F1=8,
            D=2,
            F2=16,
            kernel_length=64,
            third_kernel_size=(8, 4),
            drop_prob=dropout,
        )
    elif model_name == "eegnetv1":
        return EEGNetv1(
            n_channels,
            n_classes,
            input_window_samples=window_len_samples,
            final_conv_length=final_conv_length,
            pool_mode="max",
            second_kernel_size=(2, 32),
            third_kernel_size=(8, 4),
            drop_prob=dropout,
        )
    elif model_name == "eegresnet":
        return EEGResNet(
            n_channels,
            n_classes,
            window_len_samples,
            final_conv_length,
            n_first_filters=10,
            n_layers_per_block=2,
            first_filter_length=3,
            split_first_layer=True,
            batch_norm_alpha=0.1,
            batch_norm_epsilon=0.0001,
        )
    elif model_name == "tcn":
        return TCN(
            n_channels,
            n_classes,
            n_blocks=8,
            n_filters=2,
            kernel_size=12,
            drop_prob=dropout,
            add_log_softmax=False,
        )
    elif model_name == "sleep2020":
        return SleepStagerBlanco2020(
            n_channels,
            sampling_freq,
            n_conv_chans=20,
            input_size_s=60,
            n_classes=2,
            n_groups=3,
            max_pool_size=2,
            dropout=dropout,
            apply_batch_norm=False,
            return_feats=False,
        )
    elif model_name == "sleep2018":
        return SleepStagerChambon2018(
            n_channels,
            sampling_freq,
            n_conv_chs=8,
            time_conv_size_s=0.5,
            max_pool_size_s=0.125,
            pad_size_s=0.25,
            input_size_s=60,
            n_classes=n_classes,
            dropout=dropout,
            apply_batch_norm=False,
            return_feats=False,
        )
    elif model_name == "usleep":
        return USleep(
            in_chans=n_channels,
            sfreq=sampling_freq,
            depth=12,
            n_time_filters=5,
            complexity_factor=1.67,
            with_skip_connection=True,
            n_classes=2,
            input_size_s=60,
            time_conv_size_s=0.0703125,
            ensure_odd_conv_size=False,
            apply_softmax=False,
        )
    elif model_name == "tidnet":
        return TIDNet(
            n_channels,
            n_classes,
            window_len_samples,
            s_growth=24,
            t_filters=32,
            drop_prob=dropout,
            pooling=15,
            temp_layers=2,
            spat_layers=2,
            temp_span=0.05,
            bottleneck=3,
            summary=-1,
        )
    elif model_name == "tcn_1":
        from tcn_1 import TCN_1  # noqa: PLC0415

        return TCN_1(
            n_channels,
            n_classes,
            n_blocks=kwargs.get("tcn_n_blocks", 5),
            n_filters=kwargs.get("tcn_n_filters", 55),
            kernel_size=kwargs.get("tcn_kernel_size", 11),
            drop_prob=dropout,
            add_log_softmax=kwargs.get("tcn_add_log_softmax", True),
            input_window_samples=window_len_samples,
            last_layer_type=kwargs.get("tcn_last_layer_type", "max_pool"),
        )
    elif model_name == "hybridnet":
        return HybridNet(n_channels, n_classes, window_len_samples)
    elif model_name == "hybridnet_1":
        from hybrid_1 import HybridNet_1  # noqa: PLC0415

        return HybridNet_1(n_channels, n_classes, window_len_samples)
    elif model_name == "vit":
        from vit import ViT  # noqa: PLC0415

        return ViT(
            num_channels=n_channels,
            input_window_samples=window_len_samples,
            patch_size=kwargs.get("vit_patch_size", 10),
            num_classes=n_classes,
            dim=kwargs.get("vit_dim", 64),
            depth=kwargs.get("vit_depth", 6),
            heads=kwargs.get("vit_heads", 16),
            mlp_dim=kwargs.get("vit_mlp_dim", 64),
            dropout=dropout,
            emb_dropout=kwargs.get("vit_emb_dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")


class EEGTrainer:
    """Wraps model training and parameter persistence for an EEGClassifier."""

    def __init__(
        self,
        model: nn.Module,
        train_set,
        valid_set,
        *,
        lr: float,
        weight_decay: float,
        batch_size: int,
        n_epochs: int,
        device: str = "cpu",
        test_on_eval: bool = True,
        earlystopping: bool = False,
        early_stopping_patience_fraction: float = 1 / 3,
    ):
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.test_on_eval = test_on_eval
        self.earlystopping = earlystopping
        self.early_stopping_patience_fraction = early_stopping_patience_fraction
        self.clf: Optional[EEGClassifier] = None
        self.training_time: float = 0.0

    def _build_clf(self, weighted_loss: bool = True) -> EEGClassifier:
        monitor = lambda net: all(net.history[-1, ("train_loss_best", "valid_loss_best")])
        cp = Checkpoint(monitor=monitor, dirname="", f_criterion=None, f_optimizer=None, load_best=False)
        callbacks: list = [
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=self.n_epochs - 1)),
            ("cp", cp),
        ]
        if self.earlystopping:
            patience = max(1, int(self.n_epochs * self.early_stopping_patience_fraction))
            callbacks.append(("es", EarlyStopping(threshold=0.001, threshold_mode="rel", patience=patience)))

        criterion = (
            torch.nn.NLLLoss(weight_function(self.train_set.get_metadata().target, self.device))
            if weighted_loss
            else torch.nn.NLLLoss()
        )
        train_split = predefined_split(self.valid_set) if (weighted_loss and self.test_on_eval) else None

        return EEGClassifier(
            self.model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            train_split=train_split,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            callbacks=callbacks,
            device=self.device,
        )

    def fit(self) -> tuple[EEGClassifier, pd.DataFrame]:
        """Train the model. Returns (clf, history_df)."""
        self.clf = self._build_clf()
        torch.cuda.empty_cache()
        t0 = time.time()
        self.clf.fit(self.train_set, y=None, epochs=self.n_epochs)
        self.training_time = time.time() - t0

        results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
        history_df = pd.DataFrame(
            self.clf.history[:, results_columns],
            columns=results_columns,
            index=self.clf.history[:, "epoch"],
        ).assign(
            train_misclass=lambda df: 100 - 100 * df.train_accuracy,
            valid_misclass=lambda df: 100 - 100 * df.valid_accuracy,
        )
        return self.clf, history_df

    def save(self, path: str) -> None:
        if self.clf is None:
            raise RuntimeError("No trained classifier; call fit() first.")
        self.clf.save_params(path)

    def load(self, path: str) -> EEGClassifier:
        """Load saved parameters without training."""
        self.clf = self._build_clf(weighted_loss=False)
        self.clf.initialize()
        self.clf.load_params(path)
        return self.clf
