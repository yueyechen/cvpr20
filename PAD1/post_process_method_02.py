# coding:utf-8

import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# rootpath = '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'
list_list = []
# list_list +=[
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0210_res18_00_nir_01_4@1_result.txt',
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0210_res18_00_nir_01_4@2_result.txt',
#     '/home/users/jiachen.xue/PAD_Pytorch/work_space/result/0210_res18_00_nir_01_4@3_result.txt',
# ]
# format_seq  = ['rgb', 'depth', 'nir']
format_seq  = ['rgb']
exp_seq = ['4@1', '4@2', '4@3']
epoch_range = range(15,25)
# exp_seq = ['4@3']
list_type_seq = ['dev_frame', 'test_frame']
exp = 'res101'
work_space = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space'
save_sum = False
if save_sum:
    savepath_wo_label = os.path.join(work_space, 'result', '{}_{}_01_result_method02_wo_label_v1.txt'.format(exp, format))

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
                        prob = float(line[2])
                        if track_id in data:
                            probs = data[track_id]
                            probs.append(prob)
                        else:
                            data[track_id] = [prob]
                if list_type.startswith('dev'):
                    fw2 = open(os.path.join(work_space, 'result', exp_name, '{}_dev_e{}_02.txt'.format(exp_id, epoch)), 'w')
                else:
                    fw2 = open(os.path.join(work_space, 'result', exp_name, '{}_test_e{}_02.txt'.format(exp_id, epoch)), 'w')
                for track_id in data.keys():
                    probs = data[track_id]
                    prob_mean = np.array(probs).mean()
                    prob_var = np.var(np.array(probs))
                    prob = prob_mean*prob_var
                    write_wo = track_id + ' ' + '%.10f' % prob + '\n'
                    fw2.write(write_wo)
                    if save_sum:
                        fw_wo.write(write_wo)
                fw2.close()
if save_sum:
    fw_wo.close()