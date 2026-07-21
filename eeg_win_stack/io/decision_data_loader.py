"""Data loading and aggregation for second-stage decision models.

Handles CSV parsing from training_detail.csv (first-stage model output),
histogram generation, and patient/session-level aggregation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg_win_stack.tools.paths import findall


class DecisionDataset(Dataset):
    """PyTorch Dataset for decision-stage training.

    Holds aggregated data, labels, and valid sequence lengths.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        labels: list[int],
        valid_lens: list[int],
    ):
        """Initialize dataset.

        Parameters
        ----------
        data : list[np.ndarray]
            List of feature arrays (histogram or hybrid features).
        labels : list[int]
            List of binary labels (0 or 1).
        valid_lens : list[int]
            List of valid sequence lengths for each sample.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
        self.valid_lens = valid_lens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Return (data, label, valid_len) for index."""
        return self.data[index], self.labels[index], self.valid_lens[index]


class DecisionDataLoader:
    """Load and aggregate first-stage predictions from CSV.

    Parses training_detail.csv (output from first-stage training) and
    groups predictions into patient/session/recording aggregates.
    """

    def __init__(
        self,
        csv_path: str | Path,
        start_row: int = 1,
        n_rows: int = 4,
        row_gap: int = 4,
        block: int = 0,
    ):
        """Initialize data loader.

        Parameters
        ----------
        csv_path : str or Path
            Path to training_detail.csv from first-stage model.
        start_row : int, default=1
            Starting row index for label block.
        n_rows : int, default=4
            Number of rows containing labels.
        row_gap : int, default=4
            Gap between label block and probability block.
        block : int, default=0
            Which block of results to use (for multiple experiment blocks).
        """
        self.csv_path = Path(csv_path)
        self.start_row = start_row
        self.n_rows = n_rows
        self.row_gap = row_gap
        self.block = block

        # Parse CSV
        (
            self.labels,
            self.data,
            self.valid_lens,
            self.patients,
            self.sessions,
        ) = self._parse_csv()

    def _parse_csv(
        self,
    ) -> tuple[list[int], list[float], list[int], list[str], list[str]]:
        """Parse training_detail.csv into component arrays.

        Returns
        -------
        tuple
            (labels, data, valid_lens, patients, sessions)
        """
        pd_labels = []
        pd_valid_lens = []
        pd_data = []
        patients = []
        sessions = []

        rows_total = self.n_rows * 2 + self.row_gap
        with open(self.csv_path, newline="") as csvfile:
            results = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(results):
                start = self.start_row + self.block * rows_total
                end_labels = start + self.n_rows
                end_data = start + self.n_rows * 2
                end_valid = start + self.n_rows * 2
                end_valid_labels = start + self.n_rows * 2 + 1
                end_sessions = start + self.n_rows * 2 + 2

                if i >= start and i < end_labels:
                    pd_labels += self._remove_empty(row)
                elif i >= end_labels and i < end_data:
                    pd_data += self._remove_empty(row)
                elif i == end_valid:
                    pd_valid_lens += self._remove_empty(row)
                elif i == end_valid_labels:
                    patients += self._remove_empty(row)
                elif i == end_sessions:
                    sessions += self._remove_empty(row)

        # Add final valid length
        pd_valid_lens.append(len(pd_labels))

        # Convert types
        pd_valid_lens = [int(v) for v in pd_valid_lens]
        pd_data = [float(d) for d in pd_data]
        pd_labels = [1 if (label == "True" or label == "TRUE") else 0 for label in pd_labels]

        return pd_labels, pd_data, pd_valid_lens, patients, sessions

    @staticmethod
    def _remove_empty(row: list[str]) -> list[str]:
        """Remove empty strings from row."""
        return [x for x in row if x != ""]

    def create_histogram(
        self,
        raw: list[float],
        length: int = 10,
    ) -> np.ndarray:
        """Generate histogram from raw probabilities.

        Parameters
        ----------
        raw : list[float]
            Raw probability values.
        length : int, default=10
            Number of histogram bins.

        Returns
        -------
        np.ndarray
            Normalized histogram of shape (length,).
        """
        hist = np.zeros(length)
        for val in raw:
            bin_idx = int(val // (1 / length + 0.001))
            bin_idx = min(bin_idx, length - 1)  # Clamp to last bin
            hist[bin_idx] += 1
        return hist / (np.sum(hist) + 1e-8)  # Normalize

    def aggregate_by_criterion(
        self,
        criterion: list[str],
        length: int = 10,
        use_hybrid: bool = False,
    ) -> tuple[list[np.ndarray], list[int], list[int]]:
        """Aggregate data by criterion (patient or session).

        Parameters
        ----------
        criterion : list[str]
            List of criterion values (e.g., patient IDs or session IDs).
        length : int, default=10
            Histogram bins.
        use_hybrid : bool, default=False
            If True, concatenate histogram with padded raw data.

        Returns
        -------
        tuple
            (data_list, labels, valid_lens)
        """
        data_list = []
        labels = []
        valid_lens = []

        for criterion_val in set(criterion):
            indexes = findall(criterion, criterion_val)
            data_pa = []
            valid_len = 0

            for idx in indexes:
                data_pa += self.data[self.valid_lens[idx] : self.valid_lens[idx + 1]]
                valid_len += self.valid_lens[idx + 1] - self.valid_lens[idx]

            hist = self.create_histogram(data_pa, length)
            if use_hybrid:
                # Pad raw data to 20 samples
                padded_raw = data_pa + [0] * (20 - valid_len)
                feature = np.concatenate([hist, padded_raw])
            else:
                feature = hist

            data_list.append(feature)
            valid_lens.append(valid_len)
            labels.append(self.labels[self.valid_lens[indexes[0]]])

        return data_list, labels, valid_lens

    def load(
        self,
        aggregation: Literal["patients", "sessions", None] = None,
        length: int = 10,
        use_hybrid: bool = False,
        n_recordings: int | None = None,
    ) -> DecisionDataset:
        """Load and aggregate data into a DecisionDataset.

        Parameters
        ----------
        aggregation : {"patients", "sessions", None}, default=None
            How to group data:
            - "patients": Aggregate per patient
            - "sessions": Aggregate per session
            - None: Use per-recording data
        length : int, default=10
            Histogram bins.
        use_hybrid : bool, default=False
            Concatenate histogram with padded raw data.
        n_recordings : int or None
            If aggregation is None, limit to first n_recordings.
            If None, use all recordings.

        Returns
        -------
        DecisionDataset
            PyTorch Dataset ready for training.
        """
        data_list = []
        labels = []
        valid_lens = []

        if aggregation == "patients":
            data_list, labels, valid_lens = self.aggregate_by_criterion(
                self.patients,
                length=length,
                use_hybrid=use_hybrid,
            )
        elif aggregation == "sessions":
            # Create session identifiers
            sessions_patients = [str(p) + str(s) for p, s in zip(self.patients, self.sessions)]
            data_list, labels, valid_lens = self.aggregate_by_criterion(
                sessions_patients,
                length=length,
                use_hybrid=use_hybrid,
            )
        else:  # None: per-recording
            n_use = n_recordings if n_recordings else len(self.valid_lens) - 1
            for i in range(n_use):
                valid_len = self.valid_lens[i + 1] - self.valid_lens[i]
                valid_lens.append(valid_len)
                labels.append(self.labels[self.valid_lens[i]])

                raw_segment = self.data[self.valid_lens[i] : self.valid_lens[i + 1]]
                if use_hybrid:
                    hist = self.create_histogram(raw_segment, length=length)
                    padded_raw = raw_segment + [0] * (20 - valid_len)
                    feature = np.concatenate([hist, padded_raw])
                else:
                    feature = self.create_histogram(raw_segment, length=length)
                data_list.append(feature)

        return DecisionDataset(data_list, labels, valid_lens)

    def stats(self) -> dict:
        """Return dataset statistics.

        Returns
        -------
        dict
            Summary statistics about the parsed data.
        """
        n_recordings = len(self.valid_lens) - 1
        n_patients = len(set(self.patients))
        sessions_patients = [str(p) + str(s) for p, s in zip(self.patients, self.sessions)]
        n_sessions = len(set(sessions_patients))
        n_positive = sum(self.labels)
        pos_ratio = n_positive / len(self.labels) if self.labels else 0

        return {
            "n_recordings": n_recordings,
            "n_patients": n_patients,
            "n_sessions": n_sessions,
            "n_labels": len(self.labels),
            "n_positive": n_positive,
            "positive_ratio": pos_ratio,
        }
