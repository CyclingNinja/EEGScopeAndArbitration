"""DVC pipeline stage for second-stage decision model training and evaluation."""

from __future__ import annotations

import logging
import csv
import json
from pathlib import Path

from eeg_win_stack.api.jobs import decision_training
from eeg_win_stack.config import load

log = logging.getLogger(__name__)


def main(
    training_detail_csv: str = "training_detail.csv",
    output_dir: str = "target",
    results_csv: str = "target/decision_results.csv",
    start_row: int = 1,
    n_rows: int = 4,
    row_gap: int = 4,
    block: int = 0,
) -> None:
    """Train and evaluate second-stage decision models.

    Parameters
    ----------
    training_detail_csv : str
        Path to training_detail.csv artifact from first-stage training.
    output_dir : str
        Directory for saving models and results.
    results_csv : str
        Path to output results CSV.
    start_row : int
        Starting row for CSV parsing.
    n_rows : int
        Number of rows with labels in CSV.
    row_gap : int
        Gap between label and data blocks in CSV.
    block : int
        Which block of results to use.
    """
    config = load()
    decision_cfg = config.get("decision", {})
    training_detail_csv = decision_cfg.get("csv_path", training_detail_csv)
    results_csv = decision_cfg.get("csv_result_path", results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Training decision models from {training_detail_csv}")

    # Train
    training_results = decision_training(
        config,
        training_detail_csv_path=training_detail_csv,
        output_dir=output_dir,
        start_row=start_row,
        n_rows=n_rows,
        row_gap=row_gap,
        block=block,
        n_repetitions=config.get("decision", {}).get("n_repetitions", 1),
    )

    # Log results
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if training_results:
        fieldnames = list(training_results[0].keys())
        with results_path.open("w", newline="") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_results)

        for result in training_results:
            log.info(f"Repetition {result['repetition']}: test_acc={result['test_acc']:.4f}")

        metric_names = ("test_acc", "ori_acc", "argmax_acc", "mean_acc")
        metrics = {
            name: sum(result[name] for result in training_results) / len(training_results)
            for name in metric_names
        }
    else:
        metrics = {}

    Path("decision_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    main(*sys.argv[1:])
