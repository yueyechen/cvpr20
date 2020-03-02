import os
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

eval_list = []

def process_file(in_file, label_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    label_dict = {}
    with open(label_file, 'r') as f:
        label_inf = f.readlines()
        for val in label_inf:
            label_dict[val.split()[0]] = float(val.split()[-1])

    live_dict = dict()
    spoof_dict = dict()
    for line in lines:
        data = line.strip().split(' ')
        assert len(data) >= 2, 'wrong input'
        img_path = data[0]
        # label = float(data[-3])
        liveness = float(data[-1])
        track_id = '/'.join(img_path.split('/')[:-2])
        try:
            label = label_dict[track_id]
        except:
            print(track_id)
            print(img_path)
            print(label_dict['dev/003000'])

        if label == 1.0:
            if track_id not in live_dict:
                live_dict[track_id] = []
            live_dict[track_id].append(liveness)
        elif label == 0.0:
            if track_id not in spoof_dict:
                spoof_dict[track_id] = []
            spoof_dict[track_id].append(liveness)
        else:
            raise ValueError

    return live_dict, spoof_dict


def process_file_single(in_file, label_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
    label_dict = {}
    with open(label_file, 'r') as f:
        label_inf = f.readlines()
        for val in label_inf:
            label_dict[val.split()[0]] = float(val.split()[-1])

    live_dict = []
    spoof_dict = []
    for line in lines:
        data = line.strip().split(' ')
        if len(data) != 2:
            continue
        img_path = data[0]
        # label = float(data[-3])
        liveness = float(data[-1])
        track_id = '/'.join(img_path.split('/')[:-2])
        try:
            label = label_dict[track_id]
        except:
            print(track_id)
            print(label_dict['dev/003000'])

        if label == 1.0:
            live_dict.append(liveness)
        elif label == 0.0:
            spoof_dict.append(liveness)
        else:
            raise ValueError

    return live_dict, spoof_dict


def window_vote(config, inst_list):
    # window_length = float(config['window_length'])
    window_length = len(inst_list)
    vote_thresh = config['vote_thresh']
    thresh = config['thresh']

    inst_array = np.array(inst_list)
    index = np.where(inst_array > thresh)
    live_num = float(len(index[0]))
    if live_num / window_length >= vote_thresh:
        vote = 1.0
    else:
        vote = 0.0

    return vote

def smooth(config, inst_list):
    thresh = config['thresh']

    score = np.sum(inst_list) / float(len(inst_list))
    if score >= thresh:
        pred = 1.0
    else:
        pred = 0.0

    return pred


def evaluate(config, live_dict, spoof_dict):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for k, v in live_dict.items():
        pred = window_vote(config, v)
        # pred = smooth(config, v)
        if pred == 1.0:
            tp += 1.0
        else:
            fn += 1.0

    for k, v in spoof_dict.items():
        pred = window_vote(config, v)
        # pred = smooth(config, v)
        if pred == 0.0:
            tn += 1.0
        else:
            fp += 1.0

    TAR = float(tp) / float(tp + fn + 1e-9)
    TRR = float(tn) / float(tn + fp + 1e-9)
    APCER = float(fp) / float(tn + fp + 1e-9)
    BPCER = float(fn) / float(tp + fn + 1e-9)
    ACER = (APCER + BPCER) / 2 
    precision = float(tp + tn) / float(tp + tn + fp + fn + 1e-9)
    thresh = config['thresh']
    ignore = config.get('ignore', '0')
    if ignore == '1':
        logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(thresh, APCER * 100, BPCER * 100, ACER * 100, tp, tn, fp, fn))
    elif np.abs((thresh / 0.05) - round(thresh / 0.05)) < 1e-6:
       # logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(thresh, TAR * 100, TRR * 100, precision * 100, tp, tn, fp, fn))
        logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(thresh, APCER * 100, BPCER * 100, ACER * 100, tp, tn, fp, fn))
    # else:
    #     logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(thresh, APCER * 100, BPCER * 100, ACER * 100, tp, tn, fp, fn))
    eval_list.append([thresh, TAR, TRR, precision])

def evaluate_single_frame(config, live_list, spoof_list):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for pred in live_list:
        if pred > config['thresh']:
            tp += 1.0
        else:
            fn += 1.0

    for pred in spoof_list:
        if pred <= config['thresh']:
            tn += 1.0
        else:
            fp += 1.0

    TAR = float(tp) / float(tp + fn + 1e-9)
    TRR = float(tn) / float(tn + fp + 1e-9)
    precision = float(tp + tn) / float(tp + tn + fp + fn + 1e-9)
    thresh = config['thresh']
    if np.abs((thresh / 0.05) - round(thresh / 0.05)) < 1e-6:
        logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%'.format(config['thresh'], TAR * 100, TRR * 100, precision * 100))
    eval_list.append([config['thresh'], TAR, TRR, precision])

def evaluate_single_frame_for_select(config, live_list, spoof_list):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for pred in live_list:
        if pred > config['thresh']:
            tp += 1.0
        else:
            fn += 1.0

    for pred in spoof_list:
        if pred <= config['thresh']:
            tn += 1.0
        else:
            fp += 1.0

    TAR = float(tp) / float(tp + fn + 1e-9)
    TRR = float(tn) / float(tn + fp + 1e-9)
    precision = float(tp + tn) / float(tp + tn + fp + fn + 1e-9)
    return precision


def evaluate_for_select(config, live_dict, spoof_dict):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for k, v in live_dict.items():
        pred = window_vote(config, v)
        # pred = smooth(config, v)
        if pred == 1.0:
            tp += 1.0
        else:
            fn += 1.0

    for k, v in spoof_dict.items():
        pred = window_vote(config, v)
        # pred = smooth(config, v)
        if pred == 0.0:
            tn += 1.0
        else:
            fp += 1.0

    TAR = float(tp) / float(tp + fn + 1e-9)
    TRR = float(tn) / float(tn + fp + 1e-9)
    APCER = float(fp) / float(tn + fp + 1e-9)
    BPCER = float(fn) / float(tp + fn + 1e-9)
    ACER = (APCER + BPCER) / 2 
    
    return ACER, APCER, BPCER, tp, tn, fp, fn


def evaluate_multi_thresh(config, live_dict, spoof_dict):
    thresh_list = np.arange(0.01, 1, 0.005)
    logging.info('### Results with different thresh ###')
    # logging.info('Thresh\tTAR\tTRR\tprecision\tTP\tTN\tFP\tFN')
    logging.info('Thresh\tAPCER\tBPCER\tACER\tTP\tTN\tFP\tFN')

    for thresh in thresh_list:
        config['thresh'] = thresh
        evaluate(config, live_dict, spoof_dict)
    
def evaluate_multi_thresh_for_single_frame(config, live_list, spoof_list):
    thresh_list = np.arange(0.05, 1, 0.05)
    logging.info('### Results with different thresh ###')
    logging.info('Thresh\tTAR\tTRR\tprecision')

    for thresh in thresh_list:
        config['thresh'] = thresh
        evaluate_single_frame(config, live_list, spoof_list)

def evaluate_multi_thresh_for_single_frame_for_select(config, live_list, spoof_list):
    thresh_list = np.arange(0.05, 1, 0.05)
    # logging.info('### Results with different thresh ###')
    # logging.info('Thresh\tTAR\tTRR\tprecision')
    max_epoch_precision = 0
    max_epoch_thresh = 0

    for thresh in thresh_list:
        config['thresh'] = thresh
        precision = evaluate_single_frame_for_select(config, live_list, spoof_list)
        if precision > max_epoch_precision:
            max_epoch_precision = precision
            max_epoch_thresh = thresh
    return max_epoch_precision, max_epoch_thresh


def evaluate_single_thresh(config, live_dict, spoof_dict, thresh):
    logging.info('### Results with single thresh ###')
    # logging.info('Thresh\tTAR\tTRR\tprecision\tTP\tTN\tFP\tFN')
    logging.info('Thresh\tAPCER\tBPCER\tACER\tTP\tTN\tFP\tFN')
    config['thresh'] = thresh
    evaluate(config, live_dict, spoof_dict)

def evaluate_multi_window_thresh(config, live_dict, spoof_dict):
    thresh_list = np.arange(0.01, 1, 0.05)
    window_thresh_list = np.arange(0.001, 1, 0.0005)
    window_thresh_result = 0
    thresh_result = 0
    min_acer = 200
    min_apcer, min_bpcer, min_tp, min_tn, min_fp, min_fn = 0, 0, 0, 0, 0, 0
    for window_thresh in window_thresh_list:
        config['vote_thresh'] = window_thresh
        for thresh in thresh_list:
            config['thresh'] = thresh
            acer, apcer, bpcer, tp, tn, fp, fn = evaluate_for_select(config, live_dict, spoof_dict)
            if acer < min_acer:
                min_acer = acer
                window_thresh_result = window_thresh
                thresh_result = thresh
                min_apcer, min_bpcer, min_tp, min_tn, min_fp, min_fn = apcer, bpcer, tp, tn, fp, fn
    return min_acer, window_thresh_result, thresh_result, min_apcer, min_bpcer, min_tp, min_tn, min_fp, min_fn


def evaluate_mean(config, live_dict, spoof_dict):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for k, v in live_dict.items():
        if config['new']:
            prob = np.array(v).mean() * np.array(v).var()
        else:
            prob = np.array(v).mean()
        if prob > config['vote_thresh']:
            pred = 1.0
        else:
            pred = 0.0
        #pred = window_vote(config, v)
        # pred = smooth(config, v)
        if pred == 1.0:
            tp += 1.0
        else:
            fn += 1.0

    for k, v in spoof_dict.items():
        if config['new']:
            prob = np.array(v).mean() * np.array(v).var()
        else:
            prob = np.array(v).mean()
        if prob > config['vote_thresh']:
            pred = 1.0
        else:
            pred = 0.0
        #pred = window_vote(config, v)
        # pred = smooth(config, v)
        if pred == 0.0:
            tn += 1.0
        else:
            fp += 1.0

    TAR = float(tp) / float(tp + fn + 1e-9)
    TRR = float(tn) / float(tn + fp + 1e-9)
    APCER = float(fp) / float(tn + fp + 1e-9)
    BPCER = float(fn) / float(tp + fn + 1e-9)
    ACER = (APCER + BPCER) / 2 
    precision = float(tp + tn) / float(tp + tn + fp + fn + 1e-9)
    logging.info('Thresh\tAPCER\tBPCER\tACER\tTP\tTN\tFP\tFN')
    
    logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(0, APCER * 100, BPCER * 100, ACER * 100, tp, tn, fp, fn))
                
    


def get_TRR_by_TAR(TAR_list=[0.98, 0.95]):
    logging.info('### Results with TAR = {} ###'.format(TAR_list))
    logging.info('Thresh\tTAR\tTRR\tprecision')
    for TAR_target in TAR_list:
        TAR_gap = 9999
        result_thresh = 0
        result_TAR = 0
        result_TRR = 0
        result_precision = 0
        for result in eval_list:
            thresh = result[0]
            TAR = result[1]
            TRR = result[2]
            precision = result[3]
            if (np.abs(TAR - TAR_target) <= TAR_gap) and (TAR >= TAR_target):
                result_thresh = thresh
                result_TAR = TAR
                result_TRR = TRR
                result_precision = precision
                TAR_gap = np.abs(TAR - TAR_target)

        if result_TAR <= 1e-6 and result_TRR <= 1e-6:
            result = eval_list[0]
            result_thresh = result[0]
            result_TAR = result[1]
            result_TRR = result[2]
            result_precision = result[3]

        logging.info('{:.3f}\t{:.2f}%\t{:.2f}%\t{:.2f}%'.format(result_thresh, result_TAR * 100, result_TRR * 100, result_precision * 100))


if __name__ == '__main__':
    config = dict()
    config['window_length'] = 15  # we do not use this config setting
    config['vote_thresh'] = 0.2


    config['thresh'] = 0.5  # useless

    in_file = sys.argv[1]
    label_file = sys.argv[2]
    eval_type = sys.argv[3]
    live_dict, spoof_dict = process_file(in_file, label_file)

    # evaluate single_frames
    if eval_type == '1': 
        live_dict, spoof_dict = process_file_single(in_file, label_file)     
        evaluate_multi_thresh_for_single_frame(config, live_dict, spoof_dict)

    # evaluate muilti_frames
    elif eval_type == '2':
        config['vote_thresh'] = 0.01075269
        evaluate_multi_thresh(config, live_dict, spoof_dict)
        get_TRR_by_TAR()

    # evaluate single thresh 
    elif eval_type == '3':
        config['ignore'] = '1'
        config['vote_thresh'] = 0.01075269
        evaluate_single_thresh(config, live_dict, spoof_dict, 0.95)

    #  select best thresh and window thresh
    elif eval_type == '4':
        min_acer, window_thresh_result, thresh_result, min_apcer, min_bpcer, min_tp, min_tn, min_fp, min_fn = evaluate_multi_window_thresh(config, live_dict, spoof_dict)
        print('the window_tresh is %.6f; the thresh is %.6f' %(window_thresh_result, thresh_result))
        print('the min acer is %.6f; the min apcer is %.6f; the min bpcer is %.6f' %(min_acer, min_apcer, min_bpcer))
        print('the min tp is %.6f; the min fp is %.6f; the min tn is %.6f; the min fn is %.6f' %(min_tp, min_fp, min_tn, min_fn))
    

    # select best single frame thresh according to all test results.
    elif eval_type == '5':
        in_dirs = '/'.join(in_file.split('/')[0:-1])
        max_precision = 0
        max_precision_thresh = 0
        max_precision_epoch = 0
        for test_file in os.listdir(in_dirs):
            
            in_file_tmp = os.path.join(in_dirs, test_file)
            live_dict, spoof_dict = process_file_single(in_file_tmp, label_file)
            max_epoch_precision, max_epoch_thresh = evaluate_multi_thresh_for_single_frame_for_select(config, live_dict, spoof_dict)
            if max_epoch_precision > max_precision:
                max_precision = max_epoch_precision
                max_precision_thresh = max_epoch_thresh
                max_precision_epoch = test_file
        print('the max precision is %.6f; the coresponding thresh %.6f' %(max_precision, max_precision_thresh))
        print('the max precision coresponding epoch is %s' %(max_precision_epoch))

    elif eval_type == '6':
        config['vote_thresh'] = 0.00279075
        config['new'] = True
        evaluate_mean(config, live_dict, spoof_dict)


    else:
        print('Not implement error')
       



