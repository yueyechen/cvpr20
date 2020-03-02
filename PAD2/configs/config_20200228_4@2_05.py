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


def get_config(training=True):
    conf = edict()
    conf.model = edict()
    conf.train = edict()
    conf.eval = edict()

    # data root for training, and testing, you should change is according to your setting
    conf.data_folder = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1'

    # path for save model in training process, you should change it according to your setting
    conf.save_path = './snapshots'

    # path where training list is, you should change it according to your setting
    conf.train_list = 'anno/4@2_train.txt'

    # path where validation list is, you should change it according to your setting
    conf.val_list = 'anno/4@2_dev_test_res_label.txt'

    # path where test list is, you should change it according to your setting
    # conf.test_list = 'anno/4@2_dev_test_res_label.txt'
    conf.test_list = 'anno/phase2/4@2_test_res.txt'

    conf.batch_size = 128

    # model is saved in conf.save_path/conf.exp, if you want to train different models, you can distinguish them according to this parameter
    conf.exp = 'test-20200228-4@2-05'

    conf.model.crop_size = [256, 256]  # the crop size of our model
    conf.model.input_size = [224, 224]  # the input size of our model
    conf.model.expand_ratio = 1.0  # the expand_ratio of bbox
    conf.model.random_offset = [16, 16]  # for random crop
    conf.model.use_senet = True  # senet is adopted in our resnet18 model
    conf.model.se_reduction = 16  # parameter concerning senet
    conf.model.drop_out = 0.  # we add dropout layer in our resnet18 model
    conf.model.embedding_size = 512  # feature size of our resnet18 model

    conf.model.use_multi_color = False
    if conf.model.use_multi_color:
        conf.model.in_data_format = ['L', 'RGB', 'YCbCr']
        conf.model.in_plane = sum([color_channel_index[x]
                                   for x in conf.model.in_data_format])
    else:
        conf.model.in_data_format = 'RGB'
        conf.model.in_plane = color_channel_index[conf.model.in_data_format]

    conf.model.use_mixup = True
    conf.model.mixup_alpha = 1.0

    conf.pin_memory = True
    conf.num_workers = 4

# --------------------Training Config ------------------------
    # if training:
    conf.train.lr = 0.01  # the initial learning rate
    # epoch milestones decreased by a factor of 10
    conf.train.milestones = [20, 35, 45]
    conf.train.epoches = 50  # we trained our model for 200 epoches
    conf.train.momentum = 0.9  # parameter in setting SGD
    conf.train.weight_decay = 1e-4  # parameter in setting SGD
    conf.train.gamma = 0.1  # parameter in setting lr_scheduler

    conf.train.criterion_SL1 = nn.SmoothL1Loss()
    conf.train.criterion_ce = nn.CrossEntropyLoss()  # we use CE in training stage
    conf.train.use_center_loss = False
    conf.train.center_loss_weight = 0.01
    conf.train.criterion_cent = CenterLoss(
        num_classes=2, feat_dim=conf.model.embedding_size)

    mean_ = [0.5] * conf.model.in_plane
    std_ = [0.5] * conf.model.in_plane
    conf.train.transform = trans.Compose([  # convert input from PIL.Image to Tensor and normalized
        trans.RandomCrop(conf.model.input_size),
        # trans.RandomResizedCrop(conf.crop_size),
        trans.RandomHorizontalFlip(p=0.5),
        trans.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1)),
        trans.ToTensor(),
    ])

# --------------------Inference Config ------------------------
    conf.test = edict()
    # conf.test.save_name = '4@2-dev-single-best-acc-loss'
    conf.test.save_name = '4@2-test-single-best-acc-loss'
    conf.test.epoch = 33
    conf.test.pred_path = './result/single_frame'
    conf.test.transform = trans.Compose([  # convert input from PIL.Image to Tensor and normalized
        trans.CenterCrop(conf.model.input_size),
        trans.ToTensor(),
    ])

    return conf
