"""Tests for decision data loading (eeg_win_stack/io/decision_data_loader.py)."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from eeg_win_stack.io.decision_data_loader import (
    DecisionDataLoader,
    DecisionDataset,
)


@pytest.fixture
def sample_csv_content():
    """Sample training_detail.csv rows for testing."""
    return [
        # Labels (block 0, rows 1-4)
        ["1", "0", "1", "0"],
        ["1", "0", "1", "0"],
        ["1", "0", "1", "0"],
        ["1", "0", "1", "0"],
        # Probabilities (rows 5-8)
        ["0.1", "0.9", "0.2", "0.8"],
        ["0.2", "0.8", "0.3", "0.7"],
        ["0.15", "0.85", "0.25", "0.75"],
        ["0.18", "0.82", "0.28", "0.72"],
        # Valid lengths
        ["4", "4", "4", "4"],
        # Patients
        ["P01", "P01", "P02", "P02"],
        # Sessions
        ["S01", "S02", "S01", "S02"],
    ]


@pytest.fixture
def tmp_csv(sample_csv_content, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "training_detail.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in sample_csv_content:
            writer.writerow(row)
    return csv_file


class TestDecisionDataset:
    """Tests for DecisionDataset PyTorch wrapper."""

    @pytest.fixture
    def dataset(self):
        data = [
            np.array([0.1, 0.2, 0.15, 0.18]),
            np.array([0.9, 0.8, 0.85, 0.82]),
        ]
        labels = [0, 1]
        valid_lens = [4, 4]
        return DecisionDataset(data, labels, valid_lens)

    def test_len(self, dataset):
        assert len(dataset) == 2

    def test_getitem_returns_tuple(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 3

    def test_getitem_tensor_type(self, dataset):
        data, label, valid_len = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(valid_len, int)

    def test_getitem_values(self, dataset):
        data, label, valid_len = dataset[0]
        assert data.shape == (4,)
        assert label == 0
        assert valid_len == 4

    def test_data_dtype_float32(self, dataset):
        data, _, _ = dataset[0]
        assert data.dtype == torch.float32


class TestDecisionDataLoader:
    """Tests for DecisionDataLoader CSV parser."""

    def test_parse_csv_basic(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        assert len(loader.labels) == 4
        assert loader.labels == [1, 0, 1, 0]
        assert len(loader.data) == 4
        assert len(loader.patients) == 4
        assert len(loader.sessions) == 4

    def test_parse_csv_values(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        # Check data values are float
        assert all(isinstance(d, float) for d in loader.data)
        # Check valid lengths
        assert loader.valid_lens == [4, 4, 4, 4, 4]  # 4 recordings + 1 sentinel

    def test_parse_csv_patient_sessions(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        assert loader.patients == ["P01", "P01", "P02", "P02"]
        assert loader.sessions == ["S01", "S02", "S01", "S02"]

    def test_stats_method(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        stats = loader.stats()
        assert stats["n_recordings"] == 4
        assert stats["n_patients"] == 2
        assert stats["n_sessions"] == 3  # P01-S01, P01-S02, P02-S01/S02 (combined)
        assert stats["n_labels"] == 4
        assert stats["n_positive"] == 2
        assert abs(stats["positive_ratio"] - 0.5) < 1e-5

    def test_create_histogram_basic(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        raw = [0.1, 0.2, 0.15, 0.18, 0.25, 0.3]
        hist = loader.create_histogram(raw, length=10)

        assert hist.shape == (10,)
        assert np.allclose(hist.sum(), 1.0)  # Normalized

    def test_create_histogram_bins(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        raw = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        hist = loader.create_histogram(raw, length=10)

        assert hist.shape == (10,)
        # Each bin should have ~1 count (10 values / 10 bins), normalized to 0.1
        assert np.allclose(hist, 0.1, atol=0.05)

    def test_load_per_recording(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10)

        assert isinstance(dataset, DecisionDataset)
        assert len(dataset) == 4
        data, label, valid_len = dataset[0]
        assert data.shape == (10,)  # Histogram of length 10

    def test_load_per_recording_with_hybrid(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10, use_hybrid=True)

        data, label, valid_len = dataset[0]
        assert data.shape == (30,)  # 10 histogram + 20 padded raw

    def test_load_by_patients(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation="patients", length=10)

        # 2 unique patients
        assert len(dataset) == 2
        data, label, valid_len = dataset[0]
        assert data.shape == (10,)

    def test_load_by_sessions(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation="sessions", length=10)

        # P01-S01, P01-S02, P02-S01, P02-S02 (4 unique combinations)
        assert len(dataset) == 4

    def test_load_n_recordings_limit(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10, n_recordings=2)

        assert len(dataset) == 2

    def test_remove_empty(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        result = loader._remove_empty(["a", "", "b", "", "c"])
        assert result == ["a", "b", "c"]

    def test_aggregate_by_criterion(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        data_list, labels, valid_lens = loader.aggregate_by_criterion(
            loader.patients, length=10
        )

        assert len(data_list) == 2  # 2 unique patients
        assert len(labels) == 2
        assert len(valid_lens) == 2
        assert all(isinstance(d, np.ndarray) for d in data_list)

    def test_aggregate_with_hybrid(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        data_list, labels, valid_lens = loader.aggregate_by_criterion(
            loader.patients, length=10, use_hybrid=True
        )

        assert data_list[0].shape == (30,)  # Hybrid: 10 + 20

    def test_custom_csv_config(self, tmp_path):
        """Test with custom start_row, n_rows, row_gap."""
        csv_file = tmp_path / "custom.csv"
        rows = [
            # Empty rows 0
            [],
            # Labels (start_row=1, n_rows=2)
            ["1", "0"],
            ["1", "0"],
            # Gap (row_gap=3 means rows 3-5 are gap)
            [],
            [],
            [],
            # Probabilities (rows 6-7)
            ["0.1", "0.9"],
            ["0.2", "0.8"],
            # Valid lengths
            ["2", "2"],
            # Patients
            ["P01", "P02"],
            # Sessions
            ["S01", "S01"],
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        loader = DecisionDataLoader(
            csv_file, start_row=1, n_rows=2, row_gap=3, block=0
        )
        assert len(loader.labels) == 2
        assert len(loader.data) == 2


class TestDecisionDataIntegration:
    """Integration tests combining loader and dataset."""

    def test_end_to_end_loading(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation="patients", length=10)

        # Can iterate through dataset
        for i in range(len(dataset)):
            data, label, valid_len = dataset[i]
            assert isinstance(data, torch.Tensor)
            assert isinstance(label, int)
            assert isinstance(valid_len, int)

    def test_dataloader_integration(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10)

        batch_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False
        )
        batch_data, batch_labels, batch_lens = next(iter(batch_loader))

        assert batch_data.shape == (2, 10)
        assert len(batch_labels) == 2
        assert len(batch_lens) == 2
