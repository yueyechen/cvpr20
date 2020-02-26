# coding=utf-8
import os
import numpy as np


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
