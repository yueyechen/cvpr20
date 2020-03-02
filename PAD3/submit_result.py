import os
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from collections import OrderedDict


def process_file(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    id_dict = OrderedDict()
    for line in lines:
        data = line.strip().split(' ')
        assert len(data) >= 2, 'wrong input'
        img_path = data[0]
        # label = float(data[-3])
        liveness = float(data[-1])
        track_id = '/'.join(img_path.split('/')[:-2])
       # track_id = img_path
        if track_id not in id_dict:
            id_dict[track_id] = []
        id_dict[track_id].append(liveness)


    return id_dict


def process_file_video(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    id_dict = OrderedDict()
    for line in lines:
        data = line.strip().split(' ')
        assert len(data) >= 2, 'wrong input'
        img_path = data[0]
        # label = float(data[-3])
        liveness = float(data[-1])
       # track_id = '/'.join(img_path.split('/')[:-2])
        track_id = img_path
        if track_id not in id_dict:
            id_dict[track_id] = []
        id_dict[track_id].append(liveness)


    return id_dict

def window_vote(config, inst_list):
    window_length = len(inst_list)
    thresh = config['thresh']

    inst_array = np.array(inst_list)
    index = np.where(inst_array > thresh)
    live_num = float(len(index[0]))
    prob = live_num / window_length


    return prob

def save_result(config, id_dict, save_file):
    with open(save_file, 'w') as f:
        for k, v in id_dict.items():
            if config['mean']:
                if config['new']:
                    prob = np.array(v).mean() * np.array(v).var()
                else:
                    prob = np.array(v).mean()
            else:
                prob = window_vote(config, v)
            write_str = k + ' %.12f' % (prob,) + '\n'
            f.write(write_str)

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

def select_thresh(config, id_dict, labels):
    results = []
    for k, v in id_dict.items():
        if config['video']:
           # print('1')
            prob = v
            results.append(prob)
        elif config['mean']:
            if config['new']:
                prob = np.array(v).mean() * np.array(v).var()
            else:
                prob = np.array(v).mean()
            results.append(prob)
        else:
            prob = window_vote(config, v)
            results.append(prob)
    # print(np.array(labels).shape)
    print(np.array(results).shape)
    assert len(results) == len(labels), 'wrong, results and label must have same shape'
    bestAcc, bestThresh, APCER, NPCER, ACER, TPR_01, Thresh_at01 = evaluate_xjc(np.array(results).ravel(), np.array(labels))
    return bestThresh


if __name__ == '__main__':
    config = dict()
    config['thresh'] = 0.8375
    config['mean'] = False
    config['new'] = False
    config['video'] = False
    in_file = sys.argv[1]
    save_file = sys.argv[2]
    eval_type = sys.argv[3]
    if eval_type == '1':
        id_dict = process_file(in_file)
        save_result(config, id_dict, save_file)
    elif eval_type == '2':
        id_dict = process_file(in_file)
        labels = []
        with open(save_file, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                labels.append(float(line.split()[-1]))
        bestThresh= select_thresh(config, id_dict, labels)
        print(bestThresh)
    elif eval_type == '3':
        config['video'] = True
        id_dict = process_file_video(in_file)
        print(len(id_dict))
        labels = []
        with open(save_file, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                labels.append(float(line.split()[-2]))
        print(len(labels), labels[50])
        bestThresh= select_thresh(config, id_dict, labels)
        print(bestThresh)
    else:
        print('Not implement error')



    
