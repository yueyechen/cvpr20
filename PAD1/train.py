from config import get_config
from Learner import face_learner
import argparse
from pathlib import Path
from easydict import EasyDict as edict
from torchvision import transforms as trans


def parse_args():
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('--exp', help='training epochs', default='try', required=False, type=str)
    parser.add_argument('--train_list', help='the decay group of learning rate', default='/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_train_new.txt', required=False, type=str)
    parser.add_argument('--val_list', help='the decay group of learning rate', default='/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_dev_img_res_label_new.txt', required=False, type=str)
    parser.add_argument('--format', help='training epochs', default='rgb', required=False, type=str)
    parser.add_argument('--epochs', help='training epochs', default=25, required=False, type=int)
    parser.add_argument('--milestones', help='the decay group of learning rate', default=[10,15,20], required=False, nargs='+', type=int)
    # parser.add_argument('--milestones', help='the decay group of learning rate', default=[80,120,160], required=False, nargs='+', type=int)
    parser.add_argument('--data_path', help='the decay group of learning rate', default=Path('/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'), required=False, type=Path)
    parser.add_argument('--huoti_folder', help='the decay group of learning rate', default=Path('/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'), required=False, type=Path)
    parser.add_argument('--batch_size', help='batch size', default=128, required=False, type=int)
    parser.add_argument('--input_size', help='input size', default=[128, 128], required=False, type=int)
    parser.add_argument('--random_offset', help='input size', default=[16, 16], required=False, type=int)
    parser.add_argument('--embedding_size', help='embedding size', default=512, required=False, type=int)
    parser.add_argument('--drop_out', help='drop_out size', default=0.7, required=False, type=float)
    parser.add_argument('--use_senet', help='use_senet', default=False, required=False, type=bool)
    parser.add_argument('--se_reduction', help='se_reduction', default=16, required=False, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    conf = get_config()

    conf.train.transform = trans.Compose([
        # trans.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    exp_sequence = ['4@1', '4@2', '4@3']

    format_sequence = ['rgb']

    conf.data_path = Path('/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1')
    conf.huoti_foldfer = conf.data_path

    conf.batch_size = 128
    conf.model.input_size = [336, 336]
    conf.model.random_offset = [48, 48]
    conf.model.embedding_size = 512

    conf.train.epochs = 20
    conf.train.milestones = [8, 12, 16]
    conf.train.sampling = False
    conf.train.sampling_neg = 4

    conf.eval.input_size = [336, 336]
    conf.eval.random_offset = [48, 48]

    conf.train.gamma = 0.1
    conf.num_workers = 3
    conf.data_mode = 'huoti'
    conf.train.optim = 'sgd'

    for format in format_sequence:
        conf.train.format = format
        conf.eval.format = format
        for exp_id in exp_sequence:
            if exp_id == '4@1':
                conf.model.format = 'res50'
            elif exp_id in ['4@2', '4@3']:
                conf.model.format = 'res101'
            else:
                raise ValueError
            conf.train_list = '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/{}_train_frame_v2.txt'.format(
                exp_id)
            conf.val_list = '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/{}_dev_frame_v2.txt'.format(
                exp_id)
            conf.exp = '{}_{}_06_reg00_rect00_{}'.format(conf.model.format, format, exp_id)

            learner = face_learner(conf)
            learner.train(conf, conf.train.epochs)
            del learner