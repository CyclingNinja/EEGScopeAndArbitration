from unittest.mock import patch

from eeg_win_stack.pipeline.evaluate import configure_mlflow


def test_configure_mlflow_uses_local_experiment_when_azure_disabled():
    cfg = {
        "run": {
            "experiment_name": "test-exp",
            "use_azure_artifacts": False,
        }
    }

    with patch("eeg_win_stack.pipeline.evaluate.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = None

        configure_mlflow(cfg)

    mock_mlflow.set_tracking_uri.assert_called_once_with("mlruns")
    mock_mlflow.create_experiment.assert_called_once_with("test-exp_local")
    mock_mlflow.set_experiment.assert_called_once_with("test-exp_local")


def test_configure_mlflow_uses_azure_artifact_when_enabled():
    cfg = {
        "run": {
            "experiment_name": "test-exp",
            "use_azure_artifacts": True,
            "azure_artifact_root": "wasbs://bucket/path",
        }
    }

    with patch("eeg_win_stack.pipeline.evaluate.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = None

        configure_mlflow(cfg)

    mock_mlflow.create_experiment.assert_called_once_with(
        "test-exp",
        artifact_location="wasbs://bucket/path",
    )
