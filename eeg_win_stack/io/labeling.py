"""Label parsing and relabeling helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from eeg_win_stack.tools.paths import read_all_file_names
from eeg_win_stack.tools.logging import get_logger

logger = get_logger(__name__)


def relabel(dataset, label_path, dataset_folder):
    des = dataset.description
    des_path = list(des["path"])
    des_file = [Path(i).name for i in des_path]
    logger.debug("dataset files: %s", des_file)

    all_labelled_TUEG_file_names = []
    TUEG_labels = []

    with open(label_path, newline="") as csvfile:
        label_catalog_reader = csv.reader(csvfile, delimiter="\t")
        next(label_catalog_reader, None)

        for row in label_catalog_reader:
            if len(row) == 0:
                continue

            id_, _ = Path(row[1]).stem, None
            p_ab = float(row[2])
            label_from_ML = row[3]

            if p_ab >= 0.99 or p_ab <= 0.01:
                label = label_from_ML
            else:
                continue

            full_folder = Path(dataset_folder) / row[0][9:]
            this_file_names = read_all_file_names(str(full_folder), ".edf", key="time")
            [all_labelled_TUEG_file_names.append(ff) for ff in this_file_names if (id_ in Path(ff).name and Path(ff).name in des_file)]
            [TUEG_labels.append(label) for ff in this_file_names if (id_ in Path(ff).name and Path(ff).name in des_file)]

    logger.info("labelled %d TUEG files", len(all_labelled_TUEG_file_names))

    if "pathological" not in list(des):
        des["pathological"] = [2] * len(des["age"])

    for i in range(len(all_labelled_TUEG_file_names)):
        des["pathological"][des_file.index(Path(all_labelled_TUEG_file_names[i]).name)] = bool(int(TUEG_labels[i]))

    return des