"""Exports per-window predictions for consumption by the second-stage arbitration model."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from eeg_win_stack.tools.metrics import find_all_zero


class TrainingDetailExporter:
    """Collects per-window probabilities across all splits and writes training_detail.csv."""

    def __init__(self, clf, train_set, valid_set, test_set):
        self.clf = clf
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

    def export(self, path: str = "./training_detail.csv") -> None:
        windows_true = (
            list(self.train_set.get_metadata().target)
            + list(self.valid_set.get_metadata().target)
            + list(self.test_set.get_metadata().target)
        )
        len_true = len(windows_true)

        windows_pred = np.exp(
            np.concatenate([
                self.clf.predict_proba(self.train_set)[:, 1],
                self.clf.predict_proba(self.valid_set)[:, 1],
                self.clf.predict_proba(self.test_set)[:, 1],
            ])
        )

        len_train = len(list(self.train_set.get_metadata().target))
        len_valid_train = len(list(self.valid_set.get_metadata().target)) + len_train

        boundaries = (
            find_all_zero(self.train_set.get_metadata()["i_window_in_trial"].tolist())
            + [x + len_train for x in find_all_zero(self.valid_set.get_metadata()["i_window_in_trial"].tolist())]
            + [y + len_valid_train for y in find_all_zero(self.test_set.get_metadata()["i_window_in_trial"].tolist())]
        )

        all_paths = (
            np.array(self.train_set.description.loc[:, ["path"]]).tolist()
            + np.array(self.valid_set.description.loc[:, ["path"]]).tolist()
            + np.array(self.test_set.description.loc[:, ["path"]]).tolist()
        )
        patients = [Path(p[0]).parts[-3] for p in all_paths]
        sessions = [Path(p[0]).parts[-2] for p in all_paths]

        chunk = 16384
        with open(path, "a") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow([time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))])
            for i in range(len_true // chunk):
                writer.writerow(windows_true[i * chunk : (i + 1) * chunk])
            writer.writerow(windows_true[(len_true // chunk) * chunk :])
            for i in range(len_true // chunk):
                writer.writerow(windows_pred[i * chunk : (i + 1) * chunk])
            writer.writerow(windows_pred[(len_true // chunk) * chunk :])
            writer.writerow(boundaries)
            writer.writerow(patients)
            writer.writerow(sessions)
