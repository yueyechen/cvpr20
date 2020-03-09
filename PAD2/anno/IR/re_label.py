import os
import sys

import numpy as np


fname = sys.argv[1]


def relabel(seq):
    s = np.array(seq)
    factor = np.arange(s.shape[0]) / s.shape[0]

    return (s + factor)


lines = open(fname).readlines()
data = {}
for l in lines:
    path, *roi, label = l.strip().split()
    id_ = os.path.dirname(path)
    name = os.path.basename(path)
    if id_ in data:
        data[id_]['name'].append(name)
        data[id_]['roi'].append(roi)
        data[id_]['label'].append(int(label))
    else:
        data[id_] = {}
        data[id_]['name'] = [name]
        data[id_]['roi'] = [roi]
        data[id_]['label'] = [int(label)]


for k in sorted(data):
    if data[k]['label'][0] == 1:
        labels = relabel(data[k]['label']).tolist()
    else:
        labels = data[k]['label']

    for name, roi, label in zip(data[k]['name'], data[k]['roi'], labels):
        print('{}/{} {} {:.4f}'.format(k, name, ' '.join(roi), label))


