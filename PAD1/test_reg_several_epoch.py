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
                        default='/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_dev_img_res_label_new.txt',
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
        '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'), required=False,
                        type=Path)
    parser.add_argument('--huoti_folder', help='the decay group of learning rate', default=Path(
        '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'), required=False,
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
    # args = parse_args()
    # conf.eval.transform = trans.Compose([
    #     # trans.Resize(conf.model.input_size),
    #     trans.ToTensor(),
    #     trans.Normalize([0.5], [0.5])
    # ])

    # exp_sequence = ['4@2']
    exp_sequence = ['4@3']
    # format_sequence = ['rgb', 'depth', 'nir']
    format_sequence = ['rgb']
    epoch_range = range(8, 13)

    conf.data_path = Path('/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1')
    conf.huoti_folder = conf.data_path
    conf.result_path = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/cvpr20/work_space/result'
    # conf.result_name = 'dev.txt'

    conf.batch_size = 400
    # conf.model.input_size = [128, 128]
    # conf.model.random_offset = [76, 76]
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
        # trans.Resize(conf.model.input_size),
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for format in format_sequence:
        conf.train.format = format
        conf.eval.format = format
        for exp_id in exp_sequence:
            conf.exp = 'res101_{}_06_reg00_rect00_{}'.format(format, exp_id)
            for epoch in epoch_range:
                conf.model_path = os.path.join('/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/cvpr20/work_space/save',
                                               conf.exp,
                                               'epoch={}.pth'.format(epoch))
                for test_type in ['dev_frame', 'test_frame']:
                # for test_type in ['train_frame']:
                    conf.val_list = os.path.join('/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/{}_{}_v5.txt'.format(exp_id, test_type))
                    conf.result_name = '{}_{}_e{}.txt'.format(exp_id, test_type, epoch)

                    learner = face_learner(conf, inference=True)
                    # learner.val_train(conf)
                    # learner.val_val(conf)
                    learner.test_reg(conf)
                    del learner
    # conf.exp = args.exp
    # conf.val_list = args.val_list
    # conf.model_path = args.model_path
    # conf.result_path = args.result_path
    # conf.result_name = args.result_name
    # conf.data_folder = args.data_path
    # conf.huoti_folder = args.huoti_folder
    #
    # conf.train.format = args.format
    # conf.eval.format = args.format
    #
    # # conf.train.milestones = args.milestones
    # conf.batch_size = args.batch_size
    # conf.model.input_size = args.input_size
    # conf.model.random_offset = args.random_offset
    # conf.model.embedding_size = args.embedding_size
    # conf.model.drop_out = args.drop_out
    # conf.model.use_senet = args.use_senet
    # conf.model.se_reduction = args.se_reduction
    #
    # conf.eval.input_size = args.input_size
    # conf.eval.random_offset = args.random_offset

    # conf.train.lr = args.lr
    # # conf.batch_size = args.batch_size
    # # conf.stepsize = args.stepsize
    # conf.train.gamma = args.gamma
    # conf.num_workers = args.num_workers
    # conf.data_mode = args.data_mode
    # # conf.weight_ment = args.weight_ment
    # conf.train.milestones = args.milestones
    # # conf.note = args.note
    # conf.model.use_senet = False
    # conf.model.se_reduction = 16
    # conf.model.half_face = False
    # conf.huoti_folder = Path('/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1')
    # # conf.huoti_folder = Path('/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-SURF')
    # conf.exp = '0213_res18_00_rgb_06_4@2'
    # # conf.val_list = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@2_dev_img_res_label_new.txt'
    # conf.val_list = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@1_4@3_train_new.txt'
    # # conf.val_list = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-SURF/all_list.txt'
    # conf.eval.format = 'rgb'
    # conf.model.input_size = [336,336] # [56,112] for half face, [56,56] for quarter face, [112,112] for whole face
    # conf.model_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/save', conf.exp, 'epoch=24.pth')
    # conf.result_path = '/home/users/jiachen.xue/PAD_Pytorch/work_space/result'
    # conf.result_name = 'cross_valid_on_1_2.txt'
    # conf.result_name = 'casia-surf.txt'
    # conf.result_name = 'dev.txt'
    # conf.draw_train = True
    # conf.save_imgs = False # if save imgs depending on fp/tn/fn/tp
    # conf.draw_eval_hist = False # if draw hist on eval dataset
    # conf.draw_train_hist = False # if draw hist on train dataset
    # conf.visual = False # visualize feature maps in models
    # conf.train.transform = trans.Compose([
    #     # trans.Resize(conf.model.input_size),
    #     # trans.RandomHorizontalFlip(),
    #     trans.ToTensor(),
    #     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])
    #
    # learner = face_learner(conf, inference=True)
    #
    # # learner.val_train(conf)
    # # learner.val_val(conf)
    # learner.test(conf)
