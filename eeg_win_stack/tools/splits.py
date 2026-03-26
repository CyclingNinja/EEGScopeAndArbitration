"""Dataset splitting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split

from eeg_win_stack.tools.filters import remove_same
from eeg_win_stack.tools.paths import findall


def _split_indices_by_groups(groups, train_size, valid_size, test_size, random_state, shuffle):
    unique_groups = list(set(groups))
    idx_train_groups, idx_valid_test_groups = train_test_split(
        np.arange(len(unique_groups)),
        random_state=random_state,
        train_size=train_size,
        shuffle=shuffle,
    )
    idx_valid_groups, idx_test_groups = train_test_split(
        idx_valid_test_groups,
        random_state=random_state,
        test_size=test_size / (test_size + valid_size),
        shuffle=shuffle,
    )

    idx_train = []
    for i in idx_train_groups:
        idx_train += findall(groups, unique_groups[i])

    idx_valid = []
    for i in idx_valid_groups:
        idx_valid += findall(groups, unique_groups[i])

    idx_test = []
    for i in idx_test_groups:
        idx_test += findall(groups, unique_groups[i])

    return idx_train, idx_valid, idx_test


def split_by_proportion(windows_ds, train_size, valid_size, test_size, shuffle, random_state):
    idx_train, idx_valid_test = train_test_split(
        np.arange(len(windows_ds.description["path"])),
        random_state=random_state,
        train_size=train_size,
        shuffle=shuffle,
    )
    idx_valid, idx_test = train_test_split(
        idx_valid_test,
        random_state=random_state,
        test_size=test_size / (test_size + valid_size),
        shuffle=shuffle,
    )
    return windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})


def split_by_folder(windows_ds, train_size, valid_size, shuffle, random_state):
    des = windows_ds.description
    if "train" not in list(des):
        des["train"] = [2] * len(des["path"])

    path = des["path"]
    train = des["train"]
    for i in range(len(train)):
        if train[i] is not True and train[i] is not False:
            if "eval" in path[i]:
                des["train"][i] = False
            else:
                des["train"][i] = True

    windows_ds.set_description(des, overwrite=True)
    splits = windows_ds.split("train")
    train_valid_set = splits["True"]
    test_set = splits["False"]

    idx_train, idx_valid = train_test_split(
        np.arange(len(train_valid_set.description["path"])),
        random_state=random_state,
        train_size=train_size / (train_size + valid_size),
        shuffle=shuffle,
    )
    splits = windows_ds.split({"train": idx_train, "valid": idx_valid})
    return splits["train"], splits["valid"], test_set


def split_by_patients(windows_ds, train_size, valid_size, test_size, shuffle, random_state):
    paths = np.array(windows_ds.description.loc[:, ["path"]]).tolist()
    patients = []
    for i in range(len(paths)):
        splits = Path(paths[i][0]).parts
        patients.append(splits[-3])

    idx_train, idx_valid, idx_test = _split_indices_by_groups(
        patients, train_size, valid_size, test_size, random_state, shuffle
    )
    splits = windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
    return splits["train"], splits["valid"], splits["test"]


def split_by_sessions(windows_ds, train_size, valid_size, test_size, shuffle, random_state):
    paths = np.array(windows_ds.description.loc[:, ["path"]]).tolist()
    sessions = []
    for i in range(len(paths)):
        splits = Path(paths[i][0]).parts
        sessions.append(splits[-2] + splits[-3])

    idx_train, idx_valid, idx_test = _split_indices_by_groups(
        sessions, train_size, valid_size, test_size, random_state, shuffle
    )
    splits = windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
    return splits["train"], splits["valid"], splits["test"]


def split_train_on_tuab_tueg(
    windows_ds,
    train_size,
    valid_size,
    test_size,
    shuffle,
    random_state,
    test_on="tueg",
):
    des = windows_ds.description
    train = des["train"]
    for i in range(len(train)):
        # TODO: refactor this to use isintance instead
        if train[i] is not True and train[i] is not False:
            train[i] = "others"

    windows_ds.set_description(des, overwrite=True)
    splits = windows_ds.split("train")
    tuab_train = splits["True"]
    tuab_test = splits["False"]
    tueg_whole = splits["others"]

    paths = np.array(tueg_whole.description.loc[:, ["path"]]).tolist()
    patients = []
    for i in range(len(paths)):
        splits = Path(paths[i][0]).parts
        patients.append(splits[-3])

    unique_patients = list(set(patients))
    idx_train_patients, idx_test_patients = train_test_split(
        np.arange(len(unique_patients)),
        random_state=random_state,
        train_size=train_size + valid_size,
        shuffle=shuffle,
    )

    idx_train = []
    for i in idx_train_patients:
        idx_train += findall(patients, unique_patients[i])

    idx_test = []
    for i in idx_test_patients:
        idx_test += findall(patients, unique_patients[i])

    splits = tueg_whole.split({"train": idx_train, "test": idx_test})
    tueg_train = splits["train"]
    tueg_test = splits["test"]

    train_valid_set = BaseConcatDataset(
        ([i for i in tueg_train.datasets]) + ([j for j in tuab_train.datasets])
    )
    idx_train, idx_valid = train_test_split(
        np.arange(len(train_valid_set.description["path"])),
        random_state=random_state,
        train_size=train_size / (train_size + valid_size),
        shuffle=shuffle,
    )
    splits = train_valid_set.split({"train": idx_train, "valid": idx_valid})
    train_set = splits["train"]
    valid_set = splits["valid"]

    test_set = tueg_test if test_on == "tueg" else tuab_test
    return train_set, valid_set, test_set


def split_data(
    windows_ds,
    split_way,
    train_size,
    valid_size,
    test_size,
    shuffle,
    random_state,
    remove_attribute=None,
):
    strategies = {
        "proportion": lambda: split_by_proportion(
            windows_ds, train_size, valid_size, test_size, shuffle, random_state
        ),
        "folder": lambda: split_by_folder(
            windows_ds, train_size, valid_size, shuffle, random_state
        ),
        "patients": lambda: split_by_patients(
            windows_ds, train_size, valid_size, test_size, shuffle, random_state
        ),
        "sessions": lambda: split_by_sessions(
            windows_ds, train_size, valid_size, test_size, shuffle, random_state
        ),
        "train_on_tuab_tueg_test_on_tueg": lambda: split_train_on_tuab_tueg(
            windows_ds, train_size, valid_size, test_size, shuffle, random_state, test_on="tueg"
        ),
        "train_on_tuab_tueg_test_on_tuab": lambda: split_train_on_tuab_tueg(
            windows_ds, train_size, valid_size, test_size, shuffle, random_state, test_on="tuab"
        ),
    }

    if split_way not in strategies:
        raise ValueError(f"Unknown split_way: {split_way}")

    train_set, valid_set, test_set = strategies[split_way]()

    if remove_attribute:
        train_set = remove_same(test_set, train_set, remove_attribute)

    print("train_set:")
    print(train_set.description)
    print("valid_set:")
    print(valid_set.description)
    print("test_set:")
    print(test_set.description)
    return train_set, valid_set, test_set