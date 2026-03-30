from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from unittest.mock import Mock

from eeg_win_stack.tools.dataset_splitting import DatasetSplitter

@pytest.fixture
def windows_ds():
    ds = Mock()
    patient_1 = [f"patient_001/recording_{i:03d}" for i in range(3)]
    patient_2 = [f"patient_002/recording_{i:03d}" for i in range(3)]
    patient_3 = [f"patient_003/recording_{i:03d}" for i in range(3)]
    patient_4 = [f"patient_004/recording_{i:03d}" for i in range(3)]

    ds.description.loc = {"path": [patient_1 + patient_2 + patient_3 + patient_4]}
    ds.split.return_value = {
        "train": [0, 1, 2, 3, 4, 5],
        "valid": [6, 7, 8],
        "test": [9, 10, 11],
    }
    return ds

@pytest.fixture
def mock_dataset_splitter(windows_ds):
    ds_splitter = DatasetSplitter(windows_ds, 0.5, 0.25, 0.25, 42, False)
    return ds_splitter

def test_split_indices_by_groups_keeps_groups_together(mock_dataset_splitter):
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    idx_train, idx_valid, idx_test = mock_dataset_splitter._split_indices_by_group_labels(groups=groups)
    # Positive case: all returned indices are valid, unique, and partition the input
    all_indices = idx_train + idx_valid + idx_test
    assert sorted(all_indices) == list(range(len(groups)))
    assert len(all_indices) == len(set(all_indices))

    # Positive case: each group appears in only one split
    def group_members(indices):
        return {groups[i] for i in indices}

    train_groups = group_members(idx_train)
    valid_groups = group_members(idx_valid)
    test_groups = group_members(idx_test)

    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)

    # Positive case: indices returned for a group are exactly the positions of that group
    for split_indices in (idx_train, idx_valid, idx_test):
        for group_name in group_members(split_indices):
            expected_positions = [i for i, g in enumerate(groups) if g == group_name]
            actual_positions = [i for i in split_indices if groups[i] == group_name]
            assert actual_positions == expected_positions


def test_split_by_proportion_partitions_indices_correctly(mock_dataset_splitter, windows_ds):
    result = mock_dataset_splitter.split_by_proportion()

    windows_ds.split.assert_called_once()
    split_arg = windows_ds.split.call_args.args[0]

    assert set(split_arg.keys()) == {"train", "valid", "test"}
    assert len(split_arg["train"]) == 6
    assert len(split_arg["valid"]) == 3
    assert len(split_arg["test"]) == 3

    all_indices = list(itertools.chain.from_iterable(split_arg.values()))
    assert sorted(all_indices) == list(range(12))
    assert len(all_indices) == len(set(all_indices))

    assert result[0] == windows_ds.split.return_value['train']
    assert result[1] == windows_ds.split.return_value['valid']
    assert result[2] == windows_ds.split.return_value['test']

def test_split_by_folder_marks_eval_as_test_and_others_as_train(mock_dataset_splitter):
    pass


def test_split_by_patient(mock_dataset_splitter, windows_ds):
    result = mock_dataset_splitter.split_by_patient()

    windows_ds.split.assert_called_once()
    split_arg = windows_ds.split.call_args.args[0]

    assert set(split_arg.keys()) == {"train", "valid", "test"}
    assert len(split_arg["train"]) == 6
    assert len(split_arg["valid"]) == 3
    assert len(split_arg["test"]) == 3

    all_indices = list(itertools.chain.from_iterable(split_arg.values()))
    assert sorted(all_indices) == list(range(12))
    assert len(all_indices) == len(set(all_indices))

    def patients_in(indices):
        return {
            Path(windows_ds.description.iloc[i]["path"]).parts[-3]
            for i in indices
        }

    train_patients = patients_in(split_arg["train"])
    valid_patients = patients_in(split_arg["valid"])
    test_patients = patients_in(split_arg["test"])

    assert train_patients.isdisjoint(valid_patients)
    assert train_patients.isdisjoint(test_patients)
    assert valid_patients.isdisjoint(test_patients)

    assert len(train_patients) == 2
    assert len(valid_patients) == 1
    assert len(test_patients) == 1

    assert all(
        len([i for i in split_arg["train"] if Path(windows_ds.description.iloc[i]["path"]).parts[-3] == patient]) == 3
        for patient in train_patients
    )
    assert all(
        len([i for i in split_arg["valid"] if Path(windows_ds.description.iloc[i]["path"]).parts[-3] == patient]) == 3
        for patient in valid_patients
    )
    assert all(
        len([i for i in split_arg["test"] if Path(windows_ds.description.iloc[i]["path"]).parts[-3] == patient]) == 3
        for patient in test_patients
    )

    assert result[0] == windows_ds.split.return_value["train"]
    assert result[1] == windows_ds.split.return_value["valid"]
    assert result[2] == windows_ds.split.return_value["test"]


def test_split_by_session(windows_ds):
    pass