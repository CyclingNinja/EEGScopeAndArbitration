"""Training orchestration: builds and fits a skorch EEGClassifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from braindecode import EEGClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from eeg_win_stack.tools.metrics import weight_function


@dataclass
class TrainingConfig:
    """Hyperparameters and run settings for a single training run.

    Attributes
    ----------
    learning_rate : float
        Optimizer learning rate.
    weight_decay : float
        Optimizer weight decay (L2 regularisation strength).
    batch_size : int
        Number of windows per training batch.
    n_epochs : int
        Number of training epochs. Also sets the cosine-annealing schedule
        length (``T_max = n_epochs - 1``) and the default early-stopping
        patience (``n_epochs // 3``).
    device : str or None
        Torch device string (e.g. ``"cpu"`` or ``"cuda"``). When ``None``,
        the device is autodetected by :meth:`resolve_device`.
    early_stopping : bool
        Whether to append an :class:`~skorch.callbacks.EarlyStopping`
        callback.
    es_threshold : float
        Minimum relative improvement for early stopping to count as progress.
    es_patience : int or None
        Number of epochs without improvement before stopping. When ``None``,
        defaults to ``n_epochs // 3``.
    test_on_eval : bool
        When ``True``, the validation set is used as the skorch
        ``train_split``; when ``False``, no validation split is used.
    checkpoint_dir : str
        Directory passed to the :class:`~skorch.callbacks.Checkpoint`
        callback. Empty string uses skorch's default behaviour.
    """

    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    batch_size: int = 1
    n_epochs: int = 30
    device: str | None = None
    early_stopping: bool = True
    es_threshold: float = 0.001
    es_patience: int | None = None
    test_on_eval: bool = True
    checkpoint_dir: str = ""

    def resolve_device(self) -> str:
        """Return the device to train on, autodetecting CUDA when unset.

        Returns
        -------
        str
            :attr:`device` if explicitly set, otherwise ``"cuda"`` when a
            CUDA device is available and ``"cpu"`` otherwise.
        """
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Build, fit, and persist a braindecode class:`EEGClassifier`.

    The trainer holds only training concerns: callback assembly, classifier
    construction, the fit loop, and parameter save/load. Model creation,
    thread/cuDNN settings, and experiment logging are the caller's
    responsibility.

    Parameters
    ----------
    config : TrainingConfig, optional
        Hyperparameters and run settings. When omitted, a default
        :class:`TrainingConfig` is used.

    Attributes
    ----------
    config : TrainingConfig
        The active configuration used to build and fit classifiers.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()

    def fit(self, model, train_set, valid_set) -> EEGClassifier:
        """Train ``model`` and return the fitted classifier.

        Parameters
        ----------
        model : torch.nn.Module
            An instantiated braindecode-compatible model to train.
        train_set : braindecode.datasets.BaseConcatDataset
            Windowed dataset used for training. Its metadata targets are used
            to compute the class weights for the loss.
        valid_set : braindecode.datasets.BaseConcatDataset
            Windowed dataset used for validation when
            :attr:`TrainingConfig.test_on_eval` is ``True``.

        Returns
        -------
        EEGClassifier
            The fitted classifier. Its ``history`` attribute holds the
            per-epoch training and validation metrics.
        """
        eeg_classifier = self._build_classifier(model, train_set, valid_set)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eeg_classifier.fit(train_set, y=None, epochs=self.config.n_epochs)  # noqa
        return eeg_classifier

    def load(self, model, params_path) -> EEGClassifier:
        """Initialise a classifier around ``model`` and load saved params.

        Parameters
        ----------
        model : torch.nn.Module
            An instantiated model whose architecture matches the saved
            parameters.
        params_path : str or pathlib.Path
            Path to the ``.pt`` parameter file written by :meth:`save`.

        Returns
        -------
        EEGClassifier
            An initialised classifier with the loaded parameters, ready for
            inference.
        """
        eeg_classifier = self._build_classifier(model)
        eeg_classifier.initialize()
        eeg_classifier.load_params(str(Path(params_path)))
        return eeg_classifier

    @staticmethod
    def save(eeg_classifier, params_path) -> None:
        """Persist classifier parameters to disk.

        Parameters
        ----------
        eeg_classifier : EEGClassifier
            A fitted or initialised classifier to save.
        params_path : str or pathlib.Path
            Destination ``.pt`` file. Missing parent directories are created.
        """
        path = Path(params_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eeg_classifier.save_params(str(path))

    def _build_classifier(self, model, train_set=None, valid_set=None) -> EEGClassifier:
        """Construct an unfitted EEGClassifier from the active configuration.

        Parameters
        ----------
        model : torch.nn.Module
            The model to wrap.
        train_set : braindecode.datasets.BaseConcatDataset, optional
            When given, its targets weight the ``NLLLoss``; otherwise an
            unweighted loss is used (e.g. for inference-only loading).
        valid_set : braindecode.datasets.BaseConcatDataset, optional
            Used as the validation split when
            :attr:`TrainingConfig.test_on_eval` is ``True``.

        Returns
        -------
        EEGClassifier
            An unfitted classifier configured with the optimizer, loss,
            callbacks, and device from :attr:`config`.
        """
        cfg = self.config
        device = cfg.resolve_device()

        if train_set is not None:
            criterion = torch.nn.NLLLoss(weight_function(train_set.get_metadata().target, device))
        else:
            criterion = torch.nn.NLLLoss()

        train_split = predefined_split(valid_set) if (cfg.test_on_eval and valid_set is not None) else None

        return EEGClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            train_split=train_split,
            optimizer__lr=cfg.learning_rate,
            optimizer__weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            callbacks=self._build_callbacks(),
            device=device,
        )

    def _build_callbacks(self) -> list:
        """Assemble the skorch callback list for training.

        Returns
        -------
        list
            Callbacks for accuracy scoring, cosine-annealing LR scheduling,
            checkpointing, and (when :attr:`TrainingConfig.early_stopping`
            is enabled) early stopping.
        """
        cfg = self.config

        def monitor(net):
            return all(net.history[-1, ("train_loss_best", "valid_loss_best")])

        checkpoint = Checkpoint(
            monitor=monitor,
            dirname=cfg.checkpoint_dir,
            f_criterion=None,
            f_optimizer=None,
            load_best=False,
        )
        callbacks = [
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=cfg.n_epochs - 1)),
            ("cp", checkpoint),
        ]
        if cfg.early_stopping:
            patience = cfg.es_patience if cfg.es_patience is not None else cfg.n_epochs // 3
            early = EarlyStopping(threshold=cfg.es_threshold, threshold_mode="rel", patience=patience)
            callbacks.append(("es", early))
        return callbacks
