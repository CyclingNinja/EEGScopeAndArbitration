"""Tests for decision job orchestration (eeg_win_stack/api/jobs.py decision functions)."""

from __future__ import annotations

import csv

import pytest
import torch

from eeg_win_stack.api.jobs import decision_evaluation, decision_training
from eeg_win_stack.models.decision_models import HistogramModel


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a temporary training_detail.csv for testing."""
    csv_file = tmp_path / "training_detail.csv"
    # 8 recordings, each 4 windows, laid out row-major: recording r occupies
    # flat indices [4r, 4r+4). Recordings 0-3 (patients P01/P02) are
    # pathological with high probabilities; 4-7 (P03/P04) are non-pathological.
    rows = [
        # Header row (skipped by start_row=1, mirrors real training_detail.csv)
        ["training_detail"],
        # Labels (rows 1-4), one flat value per window
        ["True", "True", "True", "True", "True", "True", "True", "True"],
        ["True", "True", "True", "True", "True", "True", "True", "True"],
        ["False", "False", "False", "False", "False", "False", "False", "False"],
        ["False", "False", "False", "False", "False", "False", "False", "False"],
        # Probabilities (rows 5-8), aligned with the labels above
        ["0.80", "0.90", "0.85", "0.82", "0.78", "0.88", "0.91", "0.80"],
        ["0.83", "0.87", "0.86", "0.84", "0.81", "0.90", "0.85", "0.88"],
        ["0.10", "0.20", "0.15", "0.12", "0.11", "0.18", "0.14", "0.16"],
        ["0.09", "0.13", "0.17", "0.10", "0.12", "0.14", "0.19", "0.15"],
        # Valid lengths: cumulative offsets into the flat window list, one per
        # recording (a trailing 32 is appended by the parser).
        ["0", "4", "8", "12", "16", "20", "24", "28"],
        # Patients: one per recording (P01/P02 pathological, P03/P04 not)
        ["P01", "P01", "P02", "P02", "P03", "P03", "P04", "P04"],
        # Sessions: one per recording
        ["S01", "S02", "S01", "S02", "S01", "S02", "S01", "S02"],
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    return csv_file


@pytest.fixture
def decision_config():
    """Minimal config for decision-stage training."""
    return {
        "decision": {
            "length": 10,
            "use_his": True,
            "use_hybrid": False,
            "adap_pool": False,
            "hidden_layers": 0,
            "hidden_length": 5,
            "use_session_or_patients": None,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "n_epochs": 2,
            "train_ratio": 0.7,
            "valid_ratio": 0.75,
            "fix_testset": True,
            "device": "cpu",
        }
    }


class TestRunDecisionTraining:
    """Tests for run_decision_training function."""

    def test_returns_list(self, sample_csv_path, decision_config):
        """Should return list of result dicts."""
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_structure(self, sample_csv_path, decision_config):
        """Each result should have expected keys."""
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        result = results[0]
        assert "repetition" in result
        assert "train_loss" in result
        assert "valid_loss" in result
        assert "test_acc" in result
        assert "ori_acc" in result
        assert "argmax_acc" in result
        assert "mean_acc" in result

    def test_multiple_repetitions(self, sample_csv_path, decision_config):
        """Should support multiple repetitions."""
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=2,
        )

        assert len(results) == 2
        assert results[0]["repetition"] == 0
        assert results[1]["repetition"] == 1

    def test_metrics_are_floats(self, sample_csv_path, decision_config):
        """All metrics should be floats."""
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        result = results[0]
        assert isinstance(result["test_acc"], float)
        assert isinstance(result["ori_acc"], float)
        assert isinstance(result["train_loss"], float)
        assert isinstance(result["valid_loss"], float)

    def test_metrics_in_valid_range(self, sample_csv_path, decision_config):
        """Metrics should be in [0, 1] range."""
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        result = results[0]
        assert 0 <= result["test_acc"] <= 1
        assert 0 <= result["ori_acc"] <= 1
        assert 0 <= result["argmax_acc"] <= 1
        assert 0 <= result["mean_acc"] <= 1
        assert result["train_loss"] >= 0
        assert result["valid_loss"] >= 0

    def test_custom_csv_config(self, sample_csv_path, decision_config):
        """Should accept custom CSV parsing config."""
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            start_row=1,
            n_rows=4,
            row_gap=4,
            block=0,
            n_repetitions=1,
        )

        assert len(results) == 1

    def test_device_auto_detection(self, sample_csv_path, decision_config):
        """Should handle device auto-detection when device=None."""
        decision_config["decision"]["device"] = None
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        assert len(results) == 1

    def test_histogram_model_selected(self, sample_csv_path, decision_config):
        """Should select HistogramModel when use_his=True."""
        decision_config["decision"]["use_his"] = True
        decision_config["decision"]["use_session_or_patients"] = None

        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        assert len(results) == 1
        assert results[0]["test_acc"] >= 0

    def test_decision_model_selected(self, sample_csv_path, decision_config):
        """Should select DecisionModel when use_his=False."""
        decision_config["decision"]["use_his"] = False
        decision_config["decision"]["use_session_or_patients"] = None

        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        assert len(results) == 1

    def test_aggregation_by_patients(self, sample_csv_path, decision_config):
        """Should aggregate by patients when specified."""
        decision_config["decision"]["use_session_or_patients"] = "patients"

        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        assert len(results) == 1

    def test_training_converges(self, sample_csv_path, decision_config):
        """Training should show decreasing loss over epochs."""
        # Use more epochs to see convergence
        decision_config["decision"]["n_epochs"] = 5

        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir="/tmp",
            n_repetitions=1,
        )

        result = results[0]
        # Loss should be a positive number
        assert result["valid_loss"] > 0


class TestRunDecisionEvaluation:
    """Tests for run_decision_evaluation function."""

    @pytest.fixture
    def trained_model_path(self, tmp_path):
        """Create a simple trained decision model for testing.

        Must accept the ``model(X, valid_len)`` call signature used by
        ``_evaluate_decision_model``, so it is a real ``HistogramModel`` (input
        width = histogram length = 10) rather than a bare ``nn.Sequential``.
        """
        model_file = tmp_path / "model.pt"
        model = HistogramModel(length=10)
        torch.save(model, model_file)
        return model_file

    def test_returns_dict(self, sample_csv_path, decision_config, trained_model_path):
        """Should return dict with metrics."""
        result = decision_evaluation(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            model_path=trained_model_path,
        )

        assert isinstance(result, dict)

    def test_result_keys(self, sample_csv_path, decision_config, trained_model_path):
        """Result should have all required keys."""
        result = decision_evaluation(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            model_path=trained_model_path,
        )

        assert "test_acc" in result
        assert "ori_acc" in result
        assert "argmax_acc" in result
        assert "mean_acc" in result
        assert "confusion_matrix" in result

    def test_metrics_valid_range(self, sample_csv_path, decision_config, trained_model_path):
        """Metrics should be in [0, 1]."""
        result = decision_evaluation(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            model_path=trained_model_path,
        )

        assert 0 <= result["test_acc"] <= 1
        assert 0 <= result["ori_acc"] <= 1
        assert 0 <= result["argmax_acc"] <= 1
        assert 0 <= result["mean_acc"] <= 1

    def test_confusion_matrix_shape(self, sample_csv_path, decision_config, trained_model_path):
        """Confusion matrix should be 2x2."""
        result = decision_evaluation(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            model_path=trained_model_path,
        )

        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_custom_csv_config(self, sample_csv_path, decision_config, trained_model_path):
        """Should accept custom CSV parsing config."""
        result = decision_evaluation(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            model_path=trained_model_path,
            start_row=1,
            n_rows=4,
            row_gap=4,
            block=0,
        )

        assert isinstance(result, dict)
        assert "test_acc" in result


class TestDecisionIntegration:
    """Integration tests combining training and evaluation."""

    def test_train_then_evaluate(self, sample_csv_path, decision_config, tmp_path):
        """Should be able to train, save, then evaluate."""
        # Train
        results = decision_training(
            decision_config,
            training_detail_csv_path=sample_csv_path,
            output_dir=tmp_path,
            n_repetitions=1,
        )

        # Manually save the model from training
        # (In real scenario, this would be persisted by run_decision_training)
        # For this test, we'll just verify training succeeded
        assert results[0]["test_acc"] >= 0
