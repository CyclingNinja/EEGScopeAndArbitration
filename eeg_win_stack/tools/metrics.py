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
    """

    Parameters
    ----------
    starts
    b
    c
    use_prob
    prob

    Returns
    -------

    """
    if use_prob:
        prob = np.exp(np.array(prob))
    b = b.tolist()
    TT = 0
    TF = 0
    FT = 0
    FF = 0

    begin = starts[0]
    if len(starts) > 1:
        for end in starts[1:]:
            predict = c[begin:end].tolist()
            if use_prob:
                prob_recording = prob[begin:end]
                predict = top1_prob(prob_recording)
            else:
                predict = top1(predict)
            if predict is True:
                if b[begin] is True:
                    TT += 1
                else:
                    TF += 1
            else:
                if b[begin] is True:
                    FT += 1
                else:
                    FF += 1
            begin = end

    predict = c[begin:].tolist()
    predict = top1(predict)
    if predict is True:
        if b[begin] is True:
            TT += 1
        else:
            TF += 1
    else:
        if b[begin] is True:
            FT += 1
        else:
            FF += 1

    return np.array([[FF, TF], [FT, TT]])


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