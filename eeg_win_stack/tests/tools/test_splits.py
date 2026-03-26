from __future__ import annotations

import pytest

from eeg_win_stack.tools.splits import _split_indices_by_groups # noqa


def test_split_indices_by_groups_keeps_groups_together():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]

    idx_train, idx_valid, idx_test = _split_indices_by_groups(
        groups=groups,
        train_size=0.5,
        valid_size=0.25,
        test_size=0.25,
        random_state=42,
        shuffle=False,
    )

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


def test_split_indices_by_groups_raises_for_too_few_unique_groups():
    groups = ["a", "a", "b", "b"]

    with pytest.raises(ValueError):
        _split_indices_by_groups(
            groups=groups,
            train_size=0.5,
            valid_size=0.25,
            test_size=0.25,
            random_state=42,
            shuffle=False,
        )


def test_split_indices_by_groups_raises_when_split_fractions_are_invalid():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]

    with pytest.raises(ValueError):
        _split_indices_by_groups(
            groups=groups,
            train_size=0.0,
            valid_size=0.5,
            test_size=0.5,
            random_state=42,
            shuffle=False,
        )