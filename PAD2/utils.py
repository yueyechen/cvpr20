# coding=utf-8
import os

import numpy as np
import torch


class Accuracy(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_loss = 0
        self.num_inst = 0
        self.num_correct = 0

    def update(self, loss, outputs, labels):
        self.num_inst += labels.size(0)
        _, preds = torch.max(outputs, 1)
        self.num_correct += torch.sum(preds == labels.data)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def make_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
