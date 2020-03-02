# coding:utf-8

import os
import sys

import numpy as np
import prettytable as pt


def eval_for_dev(scores, labels):
    index = np.lexsort((labels, scores))
    scores = scores[index]
    labels = labels[index]

    pos = np.sum(labels)
    neg = labels.shape[0] - pos

    tp = np.cumsum(labels[::-1])[::-1]
    fp = np.cumsum((1 - labels)[::-1])[::-1]
    acc = tp + (neg - fp)
    acc = acc / (neg + pos)

    # select the last occurence
    ind_max = acc.shape[0] - np.argmax(acc[::-1]) - 1
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

    # return best_thresh, best_acc, APCER, BPCER, ACER, tp_01, thresh_at01
    return best_thresh, APCER, BPCER, ACER


def eval_for_test(scores, labels, thresh):
    pos, neg = [0, 0]
    tp, fp, tn, fn = [0, 0, 0, 0]

    for x, y in zip(scores, labels):
        x_ = 1 if x > thresh else 0
        if x_ == y:
            if y == 1:
                tp += 1
                pos += 1
            else:
                tn += 1
                neg += 1
        else:
            if y == 1:
                pos += 1
                fn += 1
            else:
                fp += 1
                neg += 1

    print('[TEST]: TP={}\tFP={}\tTN={}\tFN={}'.format(tp, fp, tn, fn))
    APCER = fp / (tn + fp)
    BPCER = fn / (fn + tp)
    ACER = (APCER + BPCER) / 2

    return APCER, BPCER, ACER


def run_dev_eval(lines):
    scores = []
    labels = []
    for idx, line in enumerate(lines):
        ctx = line.strip().split()
        labels.append(int(ctx[1]))
        scores.append(float(ctx[2]))

    res = eval_for_dev(np.array(scores), np.array(labels))

    return res


def run_test_eval(lines, thresh):
    scores = []
    labels = []

    for idx, line in enumerate(lines):
        ctx = line.strip().split()
        labels.append(int(ctx[1]))
        scores.append(float(ctx[2]))

    values = eval_for_test(scores, labels, thresh)

    return values


def run_eval(dev_file, test_file):
    lines = open(dev_file).readlines()
    dev_values = run_dev_eval(lines)
    print(
        '[DEV]-Thresh@{:.4f}\tAPCER={:.4f}\tBPCER={:.4f}\tACER={:.4f}'.format(
            *dev_values))

    lines = open(test_file).readlines()
    test_values = run_test_eval(lines, dev_values[0])
    print('[Test]-Thresh@{:.4f}: APCER={:.4f}\tBPCER={:.4f}\tACER={:.4f}\n'.format(
        dev_values[0], *test_values))


if __name__ == '__main__':
    dev_file = sys.argv[1]  # res file format: [id label score]
    test_file = sys.argv[2]

    lines = open(dev_file).readlines()
    dev_values = run_dev_eval(lines)
    print(
        '[DEV]-Thresh@{:.4f}\tAPCER={:.4f}\tBPCER={:.4f}\tACER={:.4f}'.format(
            *dev_values))

    lines = open(test_file).readlines()
    test_values = run_test_eval(lines, dev_values[0])
    print('[Test]-Thresh@{:.4f}: APCER={:.4f}\tBPCER={:.4f}\tACER={:.4f}\n'.format(
        dev_values[0], *test_values))
