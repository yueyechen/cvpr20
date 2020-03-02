# coding:utf-8

import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# rootpath = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'
# list_list = []
# list_list +=[
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0206_res18_00_4@1_result.txt',
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0206_res18_00_4@2_result.txt',
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0206_res18_00_4@3_result.txt',
# ]
listpath = '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0206_res18_00_result_method1_01_avg_w_label.txt'
# savepath_wo_label = '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0206_res18_00_result_method1_01_avg_wo_label.txt'
listpath = '/home/users/tao.cai/PAD/submit_result/test3/3.txt'


def evaluate_xjc(scores, labels):
    index = np.lexsort((labels, scores))
    scores = scores[index]
    labels = labels[index]

    pos = np.sum(labels)
    neg = labels.shape[0] - pos

    TPR = np.cumsum(labels[::-1])[::-1]
    FPR = np.cumsum((1 - labels)[::-1])[::-1]
    acc = TPR + (neg - FPR)
    acc = acc / (neg + pos)

    bestAcc = np.max(acc)
    bestThresh = np.where(acc == bestAcc)[0]
    print('number of bestThresh: %3d' % bestThresh.shape[0])
    if bestThresh.shape[0] > 1:
        bestThresh = bestThresh[-1]
    TPR_atBestThresh = TPR[bestThresh] / pos
    FPR_atBestThresh = FPR[bestThresh] / neg
    APCER = FPR[bestThresh] / neg
    NPCER = (pos - TPR[bestThresh]) / pos
    ACER = (APCER + NPCER) / 2
    bestThresh = scores[bestThresh]

    pre_TPR = TPR / pos
    pre_FPR = FPR / neg

    # TPR@FPR=10e-2
    FPR_01 = np.where(pre_FPR >= 0.01)[0][-1]
    TPR_01 = pre_TPR[FPR_01]
    Thresh_at01 = scores[FPR_01]

    return bestAcc, bestThresh, APCER, NPCER, ACER, TPR_01, Thresh_at01


if not os.path.exists(listpath):
    logging.info('{} not exist.'.format(listpath))

data_every = 200
with open(listpath, 'r')as fr:
    lines = fr.readlines()
    for idx, line in enumerate(lines):
        if idx%data_every == 0:
            scores = []
            labels = []
        line = line.strip('\n').split(' ')
        labels.append(int(line[1]))
        scores.append(float(line[2]))
        if (idx+1)%data_every == 0:
            evaluate_xjc(np.array(scores), np.array(labels))


