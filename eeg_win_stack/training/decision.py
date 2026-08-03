"""Training helpers for second-stage decision models."""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split


def split_decision_data(
    dataset,
    *,
    seed: int,
    batch_size: int,
    train_ratio: float = 0.9072,
    valid_ratio: float = 0.75,
    fix_testset: bool = True,
    test_batch_size: int = 16,
):
    """Split a decision dataset into train/valid/test DataLoaders.

    ``seed`` drives both split steps so each repetition is deterministic.
    When ``fix_testset`` is true, the train/test split keeps the test partition
    stable across repetitions by disabling shuffle on that first split.
    """

    idx_train, idx_test = train_test_split(
        torch.arange(len(dataset)),
        random_state=seed,
        train_size=train_ratio,
        shuffle=not fix_testset,
    )
    idx_train, idx_valid = train_test_split(
        idx_train,
        random_state=seed,
        train_size=valid_ratio,
        shuffle=True,
    )

    train_set = torch.utils.data.Subset(dataset, idx_train)
    valid_set = torch.utils.data.Subset(dataset, idx_valid)
    test_set = torch.utils.data.Subset(dataset, idx_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader
