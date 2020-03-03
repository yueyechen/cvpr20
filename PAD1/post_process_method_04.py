# coding:utf-8

import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# rootpath = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'
list_list = []
# list_list +=[
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0210_res18_00_nir_01_4@1_result.txt',
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0210_res18_00_nir_01_4@2_result.txt',
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0210_res18_00_nir_01_4@3_result.txt',
# ]
# format_seq  = ['rgb', 'depth', 'nir']
format_seq  = ['rgb']
# exp_seq = ['4@1', '4@2', '4@3']
exp_seq = ['4@3']
epoch_range = range(8, 13)
list_type_seq = ['dev_frame', 'test_frame']
exp = 'res101'
work_space = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space'
save_sum = False
if save_sum:
    savepath_wo_label = os.path.join(work_space, 'result', '{}_{}_01_result_method1_01_avg_wo_label_v1.txt'.format(exp, format))

if save_sum:
    if os.path.exists(savepath_wo_label):
        os.remove(savepath_wo_label)
    fw_wo = open(savepath_wo_label, 'a')

for format in format_seq:
    for exp_id in exp_seq:
        for list_type in list_type_seq:
            exp_name = '{}_{}_06_reg00_rect00_{}'.format(exp, format, exp_id)
            for epoch in epoch_range:
                listpath = os.path.join(work_space, 'result', exp_name, '{}_{}_e{}.txt'.format(exp_id, list_type, epoch))
                print(listpath)
                data = {}
                with open(listpath, 'r')as fr:
                    lines = fr.readlines()
                    for line in lines:
                        line = line.strip('\n').split(' ')
                        sec_path = line[0]
                        track_id = '/'.join(sec_path.split('/')[0:2])
                        frame_id = sec_path.split('/')[-1]
                        prob = float(line[2])
                        frame_dict = {}
                        frame_dict[frame_id] = prob
                        if track_id in data:
                            frames = data[track_id]
                            frames[frame_id] = prob
                        else:
                            data[track_id] = [frame_dict]
                if list_type.startswith('dev'):
                    fw2 = open(os.path.join(work_space, 'result', exp_name, '{}_dev_e{}_04.txt'.format(exp_id,epoch)), 'w')
                else:
                    fw2 = open(os.path.join(work_space, 'result', exp_name, '{}_test_e{}_04.txt'.format(exp_id,epoch)), 'w')
                for track_id in data.keys():
                    frames = data[track_id]
                    probs = np.array([frames[key] for key in sorted(frames.keys())])
                    num_frame = len(probs)
                    prob_mean = probs.mean()
                    prob_var = np.var(probs)
                    if num_frame<2:
                        prob_gap = 1
                    else:
                        prob_gap = probs[-(int(num_frame/2.0)):].mean() - probs[0:int(num_frame/2.0)].mean()
                    prob = prob_mean*prob_var*prob_gap
                    write_wo = track_id + ' ' + '%.10f' % prob + '\n'
                    fw2.write(write_wo)
                    if save_sum:
                        fw_wo.write(write_wo)
                fw2.close()
if save_sum:
    fw_wo.close()