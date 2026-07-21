"""Utilities for decision-stage model training and evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)


@dataclass
class DecisionTrainingResult:
    """Result from decision-stage training.

    Attributes
    ----------
    model : torch.nn.Module
        Trained model (best checkpoint from validation).
    best_valid_loss : float
        Best validation loss achieved during training.
    train_losses : list[float]
        Training loss per epoch.
    valid_losses : list[float]
        Validation loss per epoch.
    """

    model: torch.nn.Module
    best_valid_loss: float
    train_losses: list[float]
    valid_losses: list[float]


@dataclass
class DecisionEvaluationResult:
    """Result from decision-stage evaluation.

    Attributes
    ----------
    test_acc : float
        Accuracy of model predictions using argmax.
    ori_acc : float
        Accuracy of raw probability thresholding (> 0.5).
    argmax_acc : float
        Accuracy using argmax (> 50% of valid length).
    mean_acc : float
        Accuracy using mean probability vs threshold.
    confusion_matrix : np.ndarray
        2x2 confusion matrix from model predictions.
    """

    test_acc: float
    ori_acc: float
    argmax_acc: float
    mean_acc: float
    confusion_matrix: np.ndarray


def compute_decision_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    data: torch.Tensor,
    valid_lens: torch.Tensor,
    device: str = "cpu",
) -> DecisionEvaluationResult:
    """Compute all four decision evaluation metrics.

    Parameters
    ----------
    predictions : torch.Tensor
        Model logits of shape (batch_size, 2).
    targets : torch.Tensor
        True labels of shape (batch_size,).
    data : torch.Tensor
        Raw probability data of shape (batch_size, seq_len).
    valid_lens : torch.Tensor
        Valid sequence lengths of shape (batch_size,).
    device : str, default="cpu"
        Torch device.

    Returns
    -------
    DecisionEvaluationResult
        Struct containing all metrics.
    """
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    data = data.detach().cpu()
    valid_lens = valid_lens.detach().cpu()

    # Model accuracy (argmax)
    pred_labels = torch.argmax(predictions, dim=-1).numpy()
    target_labels = targets.numpy()
    test_acc = accuracy_score(target_labels, pred_labels)
    cm = confusion_matrix(target_labels, pred_labels, labels=[0, 1])

    # Raw threshold accuracy (> 0.5)
    ori_correct = 0
    ori_total = 0
    for x, y, v in zip(data, target_labels, valid_lens):
        x_binary = (x > 0.5).float()
        y_expanded = torch.full_like(x, y)
        matches = (x_binary == y_expanded)[: int(v)].sum().item()
        ori_correct += matches
        ori_total += int(v)
    ori_acc = ori_correct / ori_total if ori_total > 0 else 0

    # Argmax accuracy (> 50% of valid len)
    argmax_correct = 0
    argmax_total = 0
    for x, y, v in zip(data, target_labels, valid_lens):
        x_binary = (x > 0.5).float()
        y_expanded = torch.full_like(x, y)
        matches = (x_binary == y_expanded)[: int(v)].sum().item()
        if matches > int(v) / 2:
            argmax_correct += 1
        argmax_total += 1
    argmax_acc = argmax_correct / argmax_total if argmax_total > 0 else 0

    # Mean accuracy
    mean_correct = 0
    mean_total = 0
    for x, y, v in zip(data, target_labels, valid_lens):
        mean_prob = torch.mean(x).item()
        threshold = 0.5 * int(v) / 20
        if (mean_prob > threshold) == y:
            mean_correct += 1
        mean_total += 1
    mean_acc = mean_correct / mean_total if mean_total > 0 else 0

    return DecisionEvaluationResult(
        test_acc=test_acc,
        ori_acc=ori_acc,
        argmax_acc=argmax_acc,
        mean_acc=mean_acc,
        confusion_matrix=cm,
    )


def save_decision_results(
    result: DecisionEvaluationResult,
    output_path: str | Path,
    metadata: dict | None = None,
    append: bool = True,
) -> None:
    """Save decision evaluation results to CSV.

    Parameters
    ----------
    result : DecisionEvaluationResult
        Evaluation result to save.
    output_path : str or Path
        Path to output CSV file.
    metadata : dict or None
        Optional metadata to include as columns (e.g., config params).
    append : bool, default=True
        If True, append to existing file; if False, overwrite.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append and output_path.exists() else "w"
    with open(output_path, mode, newline="") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")

        # Write header if creating new file
        if mode == "w":
            header = [
                "test_acc",
                "ori_acc",
                "argmax_acc",
                "mean_acc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
            if metadata:
                header.extend(metadata.keys())
            writer.writerow(header)

        # Write data row
        cm = result.confusion_matrix
        row = [
            f"{result.test_acc:.6f}",
            f"{result.ori_acc:.6f}",
            f"{result.argmax_acc:.6f}",
            f"{result.mean_acc:.6f}",
            str(cm[0, 0]),  # TN
            str(cm[0, 1]),  # FP
            str(cm[1, 0]),  # FN
            str(cm[1, 1]),  # TP
        ]
        if metadata:
            row.extend([str(v) for v in metadata.values()])
        writer.writerow(row)
