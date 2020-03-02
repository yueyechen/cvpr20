import os
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

eval_list = []

def process_file(in_file, thresh):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    live_dict = dict()
    spoof_dict = dict()
    for line in lines:
        data = line.strip().split(' ')
        if len(data) < 2:
            continue
        img_path = data[0]
        label = float(data[-2])
        liveness = float(data[-1])

        if label == 1.0:
            if liveness > thresh:
                tp += 1
            else:
                fn += 1
            
        elif label == 0.0:
            if liveness > thresh:
                fp += 1
            else:
                tn += 1
        else:
            raise ValueError
    APCER = float(fp) / float(tn + fp + 1e-9)
    BPCER = float(fn) / float(tp + fn + 1e-9)
    ACER = (APCER + BPCER) / 2 

    return ACER


if __name__ == '__main__':
    input_file = sys.argv[1]
    thresh = sys.argv[2]
    acer = process_file(input_file, float(thresh))
    print(acer)


