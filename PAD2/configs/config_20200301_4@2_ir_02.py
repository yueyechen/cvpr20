import torch.nn as nn

from easydict import EasyDict as edict

from torchvision import transforms as trans
# from common.transform import MotionBlur
from nn.center_loss import CenterLoss


color_channel_index = {
    'L': 1,
    'RGB': 3,
    'YCbCr': 3,
}


def get_config():
    conf = edict()

    conf.data = edict()

    conf.model = edict()
    conf.model.network = edict()

    conf.train = edict()
    conf.train.optimizer = edict()
    conf.train.criterion = edict()

    conf.job_name = 'test-20200301-4@2-ir-02'
    conf.train_list = 'anno/IR/4@2_train.txt'
    conf.val_list = ['anno/IR/4@2_dev.txt',
                     'anno/IR/4@1_dev.txt',
                     'anno/IR/4@3_dev.txt']
    conf.test_list = ['anno/IR/4@2_dev.txt',
                      'anno/IR/4@2_test.txt']

    conf.data.folder = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'
    conf.data.crop_size = [256, 256]
    conf.data.input_size = [224, 224]
    conf.data.expand_ratio = 1.0
    conf.data.use_multi_color = False
    if conf.data.use_multi_color:
        conf.data.in_data_format = ['L', 'RGB', 'YCbCr']
        conf.data.in_plane = sum([color_channel_index[x]
                                  for x in conf.data.in_data_format])
    else:
        conf.data.in_data_format = 'RGB'
        conf.data.in_plane = color_channel_index[conf.data.in_data_format]
    conf.data.pin_memory = True
    conf.data.num_workers = 4

    conf.data.train_transform = trans.Compose([
        trans.RandomCrop(conf.data.input_size),
        # trans.RandomResizedCrop(conf.crop_size),
        trans.RandomHorizontalFlip(p=0.5),
        trans.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=(-0.1, 0.1)),
        trans.ToTensor(),
    ])
    conf.data.test_transform = trans.Compose([
        trans.CenterCrop(conf.data.input_size),
        trans.ToTensor(),
    ])

    conf.model.save_path = './snapshots'
    conf.model.batch_size = 128
    conf.model.use_mixup = True
    conf.model.mixup_alpha = 0.5
    conf.model.use_center_loss = False
    conf.model.center_loss_weight = 0.01
    conf.model.network.use_senet = True
    conf.model.network.se_reduction = 16
    conf.model.network.drop_out = 0.
    conf.model.network.embedding_size = 512

# --------------------Training Config ------------------------
    # if training:
    conf.train.epoches = 50
    conf.train.optimizer.lr = 0.001
    conf.train.optimizer.gamma = 0.1
    conf.train.optimizer.milestones = [20, 35, 45]
    conf.train.optimizer.momentum = 0.9
    conf.train.optimizer.weight_decay = 1e-4

    conf.train.criterion.sl1 = nn.SmoothL1Loss()
    conf.train.criterion.ce = nn.CrossEntropyLoss()
    conf.train.criterion.cent = CenterLoss(
        num_classes=2, feat_dim=conf.model.network.embedding_size)

    conf.test = edict()
    conf.test.save_name = ['4@2-dev-single-best-acc-loss', '4@2-test-single-best-acc-loss']
    conf.test.epochs = range(50)
    conf.test.pred_path = './result/single_frame'

    return conf
