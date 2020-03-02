import os
import sys

import numpy as np

from collections import OrderedDict as odict
from eeval import run_eval


def calc_mean_value(data):
    return np.round(np.mean(np.asarray(data)), 4)


def calc_mean_var_value(data):
    return np.mean(data) * np.var(data)


def calc_norm_value(data, ordd=2):
    return np.linalg.norm(data, ordd)


def calc_vote_value(data, thresh=0.5):
    bins = np.asarray(data) > thresh
    score = np.sum(bins) / bins.shape[0]

    return score


def generate_video_file(dirname):
    files = os.listdir(dirname)

    dev_video_files = []
    test_video_files = []

    for f in files:
        fname = os.path.join(dirname, f)
        lines = open(fname).readlines()

        res = {}

        for l in lines:
            im_name, label, score = l.strip().split()
            iid = '/'.join(im_name.split('/')[:2])
            if iid in res:
                res[iid][0].append(int(float(label)))
                res[iid][1].append(float(score))
            else:
                res[iid] = [[int(float(label))], [float(score)]]

        out_file = fname.replace('single', 'video')
        with open(out_file, 'w') as fp:
            for iid in res:
                labels, scores = res[iid]
                mean_score = calc_mean_value(scores)
                # mean_score = calc_mean_var_value(scores)
                # mean_score = calc_norm_value(scores, 2)
                # mean_score = calc_vote_value(scores, 0.5)
                ostr = '{} {} {}\n'.format(iid, labels[0], mean_score)
                fp.write(ostr)

        if 'dev-video' in out_file:
            dev_video_files.append(out_file)
        if 'test-video' in out_file:
            test_video_files.append(out_file)

    return dev_video_files, test_video_files


if __name__ == "__main__":
    dirname = sys.argv[1]
    dev_video_files, test_video_files = generate_video_file(dirname)

    for dev_file, test_file in zip(
            sorted(dev_video_files), sorted(test_video_files)):
        print('[FILE]: {}\t{}'.format(dev_file, test_file))
        run_eval(dev_file, test_file)
