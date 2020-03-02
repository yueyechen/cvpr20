from __future__ import division, print_function

import re
import sys

import numpy as np

log_file = sys.argv[1]

rows = open(log_file).read().strip()

# r = r'Val - Batch All:\s+Loss = (\d+\.\d+)\sAcc = (\d+\.\d+)'
r = r'Val#\d\s-\sBatch\sAll:\s+Loss\s=\s(\d+.\d+)\s+Acc\s=\s(\d+.\d+)'
res = re.findall(r, rows)

np_res = np.asarray(res, dtype=np.float).reshape(-1, 3, 2)
np_weight = np.array([0.2, 0.4, 0.4]).reshape(-1, 1)
np_scores = np_res * np_weight
min_ind = np.argmin(np_scores[:, :, 0].sum(axis=-1))
max_ind = np.argmax(np_scores[:, :, 1].sum(axis=-1))
ostr = 'Best Loss Epoch = {}, Value = {}\nBest Acc Epoch = {}, Value = {}'
print(ostr.format(
    min_ind, np_res[min_ind][:, 0], max_ind, np_res[max_ind][:, 1]))
