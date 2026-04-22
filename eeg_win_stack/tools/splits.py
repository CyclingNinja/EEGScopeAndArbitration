"""Dataset splitting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split

from eeg_win_stack.tools.filters import remove_same
from eeg_win_stack.tools.paths import findall
from eeg_win_stack.tools.logging import get_logger

logger = get_logger(__name__)

def split_data(windows_ds, split_way, train_size, valid_size, test_size, shuffle, random_state, remove_attribute=None):
    if split_way == "proportion":
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
        splits = windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        valid_set = splits["valid"]
        train_set = splits["train"]
        test_set = splits["test"]

    elif split_way == "folder":
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
        valid_set = splits["valid"]
        train_set = splits["train"]

    elif split_way == "patients":
        paths = np.array(windows_ds.description.loc[:, ["path"]]).tolist()
        patients = []
        sessions = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            patients.append(splits[-3])
            sessions.append(splits[-2])

        unique_patients = list(set(patients))
        idx_train_patients, idx_valid_test_patients = train_test_split(
            np.arange(len(unique_patients)),
            random_state=random_state,
            train_size=train_size,
            shuffle=shuffle,
        )
        idx_valid_patients, idx_test_patients = train_test_split(
            idx_valid_test_patients,
            random_state=random_state,
            test_size=test_size / (test_size + valid_size),
            shuffle=shuffle,
        )

        idx_train = []
        for i in idx_train_patients:
            idx_train += findall(patients, unique_patients[i])

        idx_valid = []
        for i in idx_valid_patients:
            idx_valid += findall(patients, unique_patients[i])

        idx_test = []
        for i in idx_test_patients:
            idx_test += findall(patients, unique_patients[i])

        splits = windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        valid_set = splits["valid"]
        train_set = splits["train"]
        test_set = splits["test"]

    elif split_way == "sessions":
        paths = np.array(windows_ds.description.loc[:, ["path"]]).tolist()
        patients = []
        sessions = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            patients.append(splits[-3])
            sessions.append(splits[-2] + splits[-3])

        unique_sessions = list(set(sessions))
        idx_train_patients, idx_valid_test_patients = train_test_split(
            np.arange(len(unique_sessions)),
            random_state=random_state,
            train_size=train_size,
            shuffle=shuffle,
        )
        idx_valid_patients, idx_test_patients = train_test_split(
            idx_valid_test_patients,
            random_state=random_state,
            test_size=test_size / (test_size + valid_size),
            shuffle=shuffle,
        )

        idx_train = []
        for i in idx_train_patients:
            idx_train += findall(sessions, unique_sessions[i])

        idx_valid = []
        for i in idx_valid_patients:
            idx_valid += findall(sessions, unique_sessions[i])

        idx_test = []
        for i in idx_test_patients:
            idx_test += findall(sessions, unique_sessions[i])

        splits = windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        valid_set = splits["valid"]
        train_set = splits["train"]
        test_set = splits["test"]

    elif split_way == "train_on_tuab_tueg_test_on_tueg" or split_way == "train_on_tuab_tueg_test_on_tuab":
        des = windows_ds.description
        train = des["train"]
        for i in range(len(train)):
            if train[i] is not True and train[i] is not False:
                train[i] = "others"
        windows_ds.set_description(des, overwrite=True)

        splits = windows_ds.split("train")
        tuab_train = splits["True"]
        tuab_test = splits["False"]
        tueg_whole = splits["others"]

        paths = np.array(tueg_whole.description.loc[:, ["path"]]).tolist()
        patients = []
        sessions = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            patients.append(splits[-3])
            sessions.append(splits[-2])

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

        train_valid_set = BaseConcatDataset(([i for i in tueg_train.datasets]) + ([j for j in tuab_train.datasets]))
        idx_train, idx_valid = train_test_split(
            np.arange(len(train_valid_set.description["path"])),
            random_state=random_state,
            train_size=train_size / (train_size + valid_size),
            shuffle=shuffle,
        )
        splits = train_valid_set.split({"train": idx_train, "valid": idx_valid})
        valid_set = splits["valid"]
        train_set = splits["train"]

        if split_way == "train_on_tuab_tueg_test_on_tueg":
            test_set = tueg_test
        else:
            test_set = tuab_test

    else:
        raise ValueError(f"Unknown split_way: {split_way}")

    if remove_attribute:
        train_set = remove_same(test_set, train_set, remove_attribute)

    logger.info("train_set: %s", train_set.description)
    logger.info("valid_set: %s", valid_set.description)
    logger.info("test_set: %s", test_set.description)
    return train_set, valid_set, test_set