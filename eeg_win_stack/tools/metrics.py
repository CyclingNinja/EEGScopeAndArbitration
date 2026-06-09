"""Metric and voting helpers."""

from __future__ import annotations

import numpy as np
import torch


def weight_function(targets, device="cpu"):
    """
    weighting function for datasets
    Parameters
    ----------
    targets : np.array
        A torch tensor of shape (n_samples,)
    device : str
        default 'cpu'

    Returns
    -------

    """
    weights = max(np.count_nonzero(targets == 0), np.count_nonzero(targets == 1)) / torch.tensor(
        [np.count_nonzero(targets == 0), np.count_nonzero(targets == 1)],
        dtype=torch.float,
        device=device,
    )
    return weights


def convolution_matrix(starts, b, c, use_prob=False, prob=None):
    """Recording-level confusion matrix from window-level predictions.

    Each recording spans the windows ``c[start:next_start]``; its predicted
    label is a majority vote (``top1``) or a summed-probability vote
    (``top1_prob``) when ``use_prob`` is set, and its true label is taken from
    the recording's first window.

    Parameters
    ----------
    starts : list[int]
        Window-index boundaries marking the first window of each recording.
    b : array-like
        Per-window true labels; only the value at each recording start is used.
    c : array-like
        Per-window predicted labels (majority-voting path).
    use_prob : bool
        Vote by summed class probability (``top1_prob``) instead of majority.
    prob : array-like, optional
        Per-window (log) probabilities, shape (n_windows, 2); exponentiated
        before voting. Required when ``use_prob`` is set.

    Returns
    -------
    np.ndarray
        2x2 confusion matrix laid out as ``[[FF, TF], [FT, TT]]``
        (rows = truth, cols = prediction).
    """
    if use_prob:
        prob = np.exp(np.array(prob))
    b = np.asarray(b)
    c = np.asarray(c)

    bounds = list(starts) + [len(c)]
    cm = np.zeros((2, 2), dtype=int)  # cm[true, pred]

    for begin, end in zip(bounds[:-1], bounds[1:]):
        if use_prob:
            pred = top1_prob(prob[begin:end])
        else:
            pred = bool(top1(c[begin:end].tolist()))
        true = bool(b[begin])
        cm[int(true), int(pred)] += 1

    return cm


def matthews_correlation_coefficient(con_matrix):
    """
    Mean square contingency coefficient

    Parameters
    ----------
    con_matrix : np.array
        2 x 2 matrix

    Returns
    -------

    """
    sum1 = con_matrix[0, 0] + con_matrix[0, 1]
    sum2 = con_matrix[0, 1] + con_matrix[1, 1]
    sum3 = con_matrix[1, 0] + con_matrix[1, 1]
    sum4 = con_matrix[1, 0] + con_matrix[0, 0]
    if sum1 == 0 or sum2 == 0 or sum3 == 0 or sum4 == 0:
        return 0
    return (con_matrix[0, 0] * con_matrix[1, 1] - con_matrix[1, 0] * con_matrix[0, 1]) / (
        (sum1 * sum2 * sum3 * sum4) ** 0.5
    )


def find_all_zero(input):
    return [i for i in range(len(input)) if input[i] == 0]


def top1(a_list):
    if a_list:
        return max(a_list, default="empty", key=lambda v: a_list.count(v))
    else:
        raise ValueError("Empty predictions passed to convolutional matrix")


def top1_prob(prob):
    normal = sum(prob[:, 0])
    abnormal = sum(prob[:, 1])
    return 1 if abnormal > normal else 0


def timecost(time_duration):
    m, s = divmod(time_duration, 60)
    h, m = divmod(m, 60)
    return "%dh:%dm:%ds" % (h, m, s)