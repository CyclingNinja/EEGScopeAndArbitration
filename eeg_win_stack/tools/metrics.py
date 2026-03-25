"""Metric and voting helpers."""

from __future__ import annotations

import numpy as np
import torch


def weight_function(targets, device="cpu"):
    weights = max(np.count_nonzero(targets == 0), np.count_nonzero(targets == 1)) / torch.tensor(
        [np.count_nonzero(targets == 0), np.count_nonzero(targets == 1)],
        dtype=torch.float,
        device=device,
    )
    return weights


def matthews_correlation_coefficient(con_matrix):
    sum1 = con_matrix[0, 0] + con_matrix[0, 1]
    sum2 = con_matrix[0, 1] + con_matrix[1, 1]
    sum3 = con_matrix[1, 1] + con_matrix[1, 0]
    sum4 = con_matrix[1, 0] + con_matrix[0, 0]
    if sum1 == 0 or sum2 == 0 or sum3 == 0 or sum4 == 0:
        return 0
    return (con_matrix[0, 0] * con_matrix[1, 1] - con_matrix[1, 0] * con_matrix[0, 1]) / (
        (sum1 * sum2 * sum3 * sum4) ** 0.5
    )


def find_all_zero(input):
    return [i for i in range(len(input)) if input[i] == 0]


def top1(lst):
    return max(lst, default="empty", key=lambda v: lst.count(v))


def top1_prob(prob):
    normal = sum(prob[:, 0])
    abnormal = sum(prob[:, 1])
    return 1 if abnormal > normal else 0


def top1_prob1(prob, predict):
    abnormal = sum(prob[:, 1] * predict)
    predict = 1 - np.array(predict)
    normal = sum(prob[:, 0] * predict)
    return 1 if abnormal > normal else 0


def con_mat(starts, b, c, use_prob=False, prob=None):
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


def timecost(time_dutation):
    m, s = divmod(time_dutation, 60)
    h, m = divmod(m, 60)
    return "%dh:%dm:%ds" % (h, m, s)