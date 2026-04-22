"""Dataset filtering helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_win_stack.tools.paths import get_full_filelist
from eeg_win_stack.tools.logging import get_logger

logger = get_logger(__name__)


def remove_tuab_from_dataset(ds, tuab_loc):
    tuab_list = get_full_filelist(tuab_loc, ".edf")
    tuab_list = [Path(path).name for path in tuab_list]

    split_ids = []
    for d_i, d in enumerate(ds.description["path"]):
        file_name = Path(d).name
        if file_name not in tuab_list:
            split_ids.append(d_i)

    splits = ds.split(split_ids)
    return splits["0"]


def remove_same(ds1, ds2, attribute):
    remove_num = 0
    if attribute == "file_name":
        loc = -1
    elif attribute == "patients":
        loc = -3
    elif attribute == "sessions":
        loc = -2
    else:
        raise ValueError(f"Unknown attribute: {attribute}")

    paths = np.array(ds1.description.loc[:, ["path"]]).tolist()
    attributes = []
    for i in range(len(paths)):
        splits = Path(paths[i][0]).parts
        attributes.append(splits[loc])

    unique_attributes = list(set(attributes))
    split_ids = []
    for d_i, d in enumerate(ds2.description["path"]):
        attributes2 = Path(d).parts[loc]
        if attributes2 not in unique_attributes:
            split_ids.append(d_i)
        else:
            remove_num += 1

    logger.info("removed %d recordings sharing attribute with test set", remove_num)
    splits = ds2.split(split_ids)
    return splits["0"]


def select_by_duration(ds, tmin=0, tmax=None):
    if tmax is None:
        tmax = np.inf

    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        duration = d.raw.n_times / d.raw.info["sfreq"]
        if tmin <= duration <= tmax:
            split_ids.append(d_i)

    splits = ds.split(split_ids)
    return splits["0"]


def select_labeled(ds):
    split_ids = []
    for d_i, d in enumerate(ds.description["pathological"]):
        if d is True or d is False:
            split_ids.append(d_i)

    splits = ds.split(split_ids)
    return splits["0"]


def exclude_by_name(ds, names):
    split_ids = []
    for d_i, d in enumerate(ds.description["path"]):
        if Path(d).name not in names:
            split_ids.append(d_i)
        else:
            logger.debug("excluding overlapping recording: %s", Path(d).name)

    splits = ds.split(split_ids)
    return splits["0"]


def select_by_channel(ds, channels):
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        include = True
        for chan in channels:
            if chan not in d.raw.info["ch_names"]:
                include = False
                break
        if include:
            split_ids.append(d_i)

    splits = ds.split(split_ids)
    return splits["0"]


def check_inf(ds):
    for d_i, d in enumerate(ds.datasets):
        logger.debug("recording %d info: %s", d_i, d.raw.info)