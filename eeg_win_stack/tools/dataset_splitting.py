"""Dataset splitting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split

from eeg_win_stack.tools.filters import remove_same
from eeg_win_stack.tools.paths import findall


class DatasetSplitter:
    def __init__(
        self,
        windows_ds,
        train_size,
        valid_size,
        test_size,
        random_state,
        shuffle=True,
        remove_attribute=None,
    ):
        """

        Parameters
        ----------
        windows_ds : BaseConcatDataset
        train_size : float
        valid_size : float
        test_size : float
        random_state : list
        shuffle : bool, optional
        remove_attribute : str, optional
        """
        self.windows_ds = windows_ds
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.remove_attribute = remove_attribute

    def _remove_attribute_check(self, train_set, test_set):
        if self.remove_attribute:
            return remove_same(test_set, train_set, self.remove_attribute)
        return train_set

    def _split_indices_by_group_labels(self, groups):
        unique_groups = list(set(groups))
        idx_train_groups, idx_valid_test_groups = train_test_split(
            np.arange(len(unique_groups)),
            random_state=self.random_state,
            test_size=self.train_size,
            shuffle=self.shuffle,
        )
        idx_valid_groups, idx_test_groups = train_test_split(
            idx_valid_test_groups,
            random_state=self.random_state,
            test_size=self.test_size / (self.test_size + self.valid_size),
            shuffle=self.shuffle,
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

    def split_by_proportion(self):
        idx_train, idx_valid_test = train_test_split(
            np.arange(len(self.windows_ds.description["path"])),
            random_state=self.random_state,
            train_size=self.train_size,
            shuffle=self.shuffle,
        )
        idx_valid, idx_test = train_test_split(
            idx_valid_test,
            random_state=self.random_state,
            test_size=self.test_size / (self.test_size + self.valid_size),
            shuffle=self.shuffle,
        )
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        return splits["train"], splits["valid"], splits["test"]

    def split_by_folder(self):
        des = self.windows_ds.description
        if "train" not in list(des):
            des["train"] = [2] * len(des["path"])

        path = des["path"]
        train = des["train"]
        for i in range(len(train)):
            if isinstance(train[i], bool):
                des["train"][i] = "eval" not in path[i]

        self.windows_ds.set_description(des, overwrite=True)
        splits = self.windows_ds.split("train")
        train_valid_set = splits["True"]
        test_set = splits["False"]

        idx_train, idx_valid = train_test_split(
            np.arange(len(train_valid_set.description["path"])),
            random_state=self.random_state,
            train_size=self.train_size / (self.train_size + self.valid_size),
            shuffle=self.shuffle,
        )
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid})
        train_set = splits["train"]
        valid_set = splits["valid"]

        return train_set, valid_set, test_set

    def split_by_patient(self):
        paths = np.array(self.windows_ds.description.loc[:, ["path"]]).tolist()
        patients = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            patients.append(splits[-3])

        idx_train, idx_valid, idx_test = self._split_indices_by_group_labels(patients)
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        return splits["train"], splits["valid"], splits["test"]

    def split_by_session(self):
        paths = np.array(self.windows_ds.description.loc[:, ["path"]]).tolist()
        sessions = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            sessions.append(splits[-2] + splits[-3])

        idx_train, idx_valid, idx_test = self._split_indices_by_group_labels(sessions)
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        return splits["train"], splits["valid"], splits["test"]

    def split_tuab_tueg(self, test_on="tueg"):
        des = self.windows_ds.description
        train = des["train"]
        for i in range(len(train)):
            if isinstance(train[i], bool):
                train[i] = "others"

        self.windows_ds.set_description(des, overwrite=True)
        splits = self.windows_ds.split("train")
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
            random_state=self.random_state,
            train_size=self.train_size + self.valid_size,
            shuffle=self.shuffle,
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

        train_valid_set = BaseConcatDataset((([i for i in tueg_train.datasets]) + ([j for j in tuab_train.datasets])))
        idx_train, idx_valid = train_test_split(
            np.arange(len(train_valid_set.description["path"])),
            random_state=self.random_state,
            train_size=self.train_size / (self.train_size + self.valid_size),
            shuffle=self.shuffle,
        )
        splits = train_valid_set.split({"train": idx_train, "valid": idx_valid})
        train_set = splits["train"]
        valid_set = splits["valid"]

        test_set = tueg_test if test_on == "tueg" else tuab_test
        return train_set, valid_set, test_set

    def split_data(self, split_way):
        strategies = {
            "proportion": self.split_by_proportion,
            "folder": self.split_by_folder,
            "patients": self.split_by_patient,
            "sessions": self.split_by_session,
            "train_on_tuab_tueg_test_on_tueg": lambda: self.split_tuab_tueg(test_on="tueg"),
            "train_on_tuab_tueg_test_on_tuab": lambda: self.split_tuab_tueg(test_on="tuab"),
        }

        if split_way not in strategies:
            raise ValueError(f"Unknown split_way: {split_way}")

        train_set, valid_set, test_set = strategies[split_way]()
        train_set = self._remove_attribute_check(train_set, test_set)

        print("train_set:")
        print(train_set.description)
        print("valid_set:")
        print(valid_set.description)
        print("test_set:")
        print(test_set.description)
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
    splitter = DatasetSplitter(
        windows_ds=windows_ds,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        remove_attribute=remove_attribute,
    )
    return splitter.split_data(split_way)