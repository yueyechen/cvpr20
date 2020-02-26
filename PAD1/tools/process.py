import sys

import numpy as np


fname = sys.argv[1]

lines = open(fname).readlines()

res = {}

for l in lines:
    im_name, label = l.strip().split()
    iid = '/'.join(im_name.split('/')[:2])
    if iid in res:
        res[iid].append(float(label))
    else:
        res[iid] = [float(label)]


def calc_avg_value(data):
    return np.round(np.mean(np.asarray(data)), 4)

for iid in sorted(res):
    num = int(iid.split('/')[-1])
    label = 1 if num % 2 == 0 else 0
    value = calc_avg_value(res[iid])
    print(iid, label, value)

