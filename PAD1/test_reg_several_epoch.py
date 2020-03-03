from config import get_config
from Learner import face_learner
from pathlib import Path
import argparse
from torchvision import transforms as trans
from easydict import EasyDict as edict
import os

def parse_args():
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('--exp', help='training epochs', default='try', required=False, type=str)
    parser.add_argument('--val_list', help='the decay group of learning rate',
                        default='/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_dev_img_res_label_new.txt',
                        required=False, type=str)
    parser.add_argument('--model_path', help='the decay group of learning rate',
                        default='/home/users/jiachen.xue/PAD_Pytorch/work_space/save/0213_res18_00_rgb_05_4@3/epoch=24.pth',
                        required=False, type=str)
    parser.add_argument('--result_path', help='the decay group of learning rate',
                        default='/home/users/jiachen.xue/PAD_Pytorch/work_space/result',
                        required=False, type=str)
    parser.add_argument('--result_name', help='the decay group of learning rate',
                        default='dev_new.txt',
                        required=False, type=str)
    parser.add_argument('--format', help='training epochs', default='rgb', required=False, type=str)
    # parser.add_argument('--epochs', help='training epochs', default=25, required=False, type=int)
    # parser.add_argument('--milestones', help='the decay group of learning rate', default=[10, 15, 20], required=False,
    #                     type=list)
    parser.add_argument('--data_path', help='the decay group of learning rate', default=Path(
        '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'), required=False,
                        type=Path)
    parser.add_argument('--huoti_folder', help='the decay group of learning rate', default=Path(
        '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'), required=False,
                        type=Path)
    parser.add_argument('--batch_size', help='batch size', default=128, required=False, type=int)
    parser.add_argument('--input_size', help='input size', default=[128, 128], required=False, type=list)
    parser.add_argument('--random_offset', help='input size', default=[16, 16], required=False, type=list)
    parser.add_argument('--embedding_size', help='embedding size', default=512, required=False, type=int)
    parser.add_argument('--drop_out', help='drop_out size', default=0.7, required=False, type=float)
    parser.add_argument('--use_senet', help='use_senet', default=False, required=False, type=bool)
    parser.add_argument('--se_reduction', help='se_reduction', default=16, required=False, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    conf = get_config()
    exp_sequence = ['4@3']
    format_sequence = ['rgb']
    epoch_range = range(8, 13)

    conf.data_path = Path('/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1')
    conf.huoti_folder = conf.data_path
    conf.result_path = '/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result'

    conf.batch_size = 400
    conf.model.embedding_size = 512
    conf.model.drop_out = 0.7
    conf.model.use_senet = False
    conf.model.se_reduction = 16

    conf.train.epochs = 25
    conf.train.milestones = [10, 15, 20]
    conf.train.sampling = False
    conf.train.sampling_neg = 4
    conf.train.rectified = True

    conf.eval.input_size = [336, 336]
    conf.eval.random_offset = [48, 48]

    conf.train.gamma = 0.1
    conf.num_workers = 3
    conf.data_mode = 'huoti'
    conf.train.optim = 'sgd'

    conf.save_imgs = False  # if save imgs depending on fp/tn/fn/tp
    conf.draw_eval_hist = False  # if draw hist on eval dataset
    conf.draw_train_hist = False  # if draw hist on train dataset
    conf.visual = False  # visualize feature maps in models
    conf.train.transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for format in format_sequence:
        conf.train.format = format
        conf.eval.format = format
        for exp_id in exp_sequence:
            conf.exp = 'res101_{}_06_reg00_rect00_{}'.format(format, exp_id)
            for epoch in epoch_range:
                conf.model_path = os.path.join('/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/save',
                                               conf.exp,
                                               'epoch={}.pth'.format(epoch))
                for test_type in ['dev_frame', 'test_frame']:
                    conf.val_list = os.path.join('/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/{}_{}_v5.txt'.format(exp_id, test_type))
                    conf.result_name = '{}_{}_e{}.txt'.format(exp_id, test_type, epoch)

                    learner = face_learner(conf, inference=True)
                    learner.test_reg(conf)
                    del learner