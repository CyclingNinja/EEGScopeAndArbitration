"""EEG model evaluation."""

from __future__ import annotations

import csv
import time
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix

from eeg_win_stack.tools.metrics import (
    con_mat,
    find_all_zero,
    matthews_correlation_coefficient,
)


class EEGEvaluator:
    """Computes per-window and per-recording metrics and writes results to CSV."""

    def __init__(self, clf, test_set):
        self.clf = clf
        self.test_set = test_set

    def evaluate(self) -> dict[str, Any]:
        """Run predictions and compute metrics. Returns a metrics dict."""
        y_true = self.test_set.get_metadata().target
        y_pred = self.clf.predict(self.test_set)
        y_pred_proba = self.clf.predict_proba(self.test_set)

        starts = find_all_zero(self.test_set.get_metadata()["i_window_in_trial"].tolist())
        window_cm = confusion_matrix(y_true, y_pred)
        recording_cm = con_mat(starts, y_true, y_pred)

        def _metrics(cm):
            return {
                "precision": cm[0, 0] / (cm[0, 0] + cm[1, 0]),
                "recall": cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                "acc": (cm[0, 0] + cm[1, 1]) / cm.sum(),
                "mcc": matthews_correlation_coefficient(cm),
            }

        wm = _metrics(window_cm)
        rm = _metrics(recording_cm)

        return {
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "confusion_mat": window_cm,
            "confusion_mat_per_recording": recording_cm,
            "acc": wm["acc"],
            "precision": wm["precision"],
            "recall": wm["recall"],
            "mcc": wm["mcc"],
            "precision_per_recording": rm["precision"],
            "recall_per_recording": rm["recall"],
            "acc_per_recording": rm["acc"],
            "mcc_per_recording": rm["mcc"],
        }

    @staticmethod
    def write_header(log_path: str, columns: list[str]) -> None:
        with open(log_path, "a") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow([time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))])
            writer.writerow(columns)

    def write_results(
        self,
        log_path: str,
        row_values: list,
        history_df: pd.DataFrame | None = None,
    ) -> None:
        """Write per-epoch history rows (if provided) then the full result row."""
        with open(log_path, "a") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            if history_df is not None:
                for epoch_idx in range(len(history_df) - 1):
                    row = history_df.iloc[epoch_idx]
                    writer.writerow(
                        [row["train_loss"], row["valid_loss"], row["train_accuracy"], row["valid_accuracy"]]
                    )
            writer.writerow(row_values)
