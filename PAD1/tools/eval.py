# coding:utf-8

import os
import sys

import numpy as np
import prettytable as pt


fname = sys.argv[1]  # res file format: [id label score]

NUM_PER_FILE = 200



def evaluate(scores, labels):
    index = np.lexsort((labels, scores))
    scores = scores[index]
    labels = labels[index]

    pos = np.sum(labels)
    neg = labels.shape[0] - pos

    tp = np.cumsum(labels[::-1])[::-1]
    fp = np.cumsum((1 - labels)[::-1])[::-1]
    acc = tp + (neg - fp)
    acc = acc / (neg + pos)

    ind_max = acc.shape[0] - np.argmax(acc[::-1]) - 1  # select the last occurence
    value_max = acc[ind_max]
    best_thresh = scores[ind_max]
    best_acc = acc[ind_max]

    best_tp = tp[ind_max] / pos
    best_fp = fp[ind_max] / neg

    APCER = fp[ind_max] / neg
    BPCER = (pos - tp[ind_max]) / pos
    ACER = (APCER + BPCER) / 2

    pre_tp = tp / pos
    pre_fp = fp / neg

    # tp@fp=10e-2
    fp_01 = np.where(pre_fp >= 0.01)[0][-1]
    tp_01 = pre_tp[fp_01]
    thresh_at01 = scores[fp_01]

    return best_thresh, best_acc, APCER, BPCER, ACER, tp_01, thresh_at01


if __name__ == '__main__':
    tb = pt.PrettyTable()
    tb.field_names = ["No", "Thresh", "Accuracy", "APCER", "BPCER", "ACER", "TPR@1e-2", "Thresh@1e-2"]

    lines = open(fname).readlines()

    for idx, line in enumerate(lines):
        if idx % NUM_PER_FILE == 0:
            scores = []
            labels = []

        ctx = line.strip().split()
        labels.append(int(ctx[1]))
        scores.append(float(ctx[2]))

        out_str = 'No\tThresh\tAccuracy\tAPCER\tBPCER\tACER\tTPR@1e-2\tThresh@1e-2'.format(idx)

        if (idx + 1) % NUM_PER_FILE == 0:
            res = evaluate(np.array(scores), np.array(labels))
            tb.add_row([(idx + 1) / NUM_PER_FILE] + (list(res)))
            print(tb)

