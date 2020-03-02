from easydict import EasyDict as edict
from pathlib import Path
import torch
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1')
    conf.huoti_folder = conf.data_path
    conf.train_list = '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_train_new.txt'

    conf.work_path = Path('/home/users/jiachen.xue/anti_spoofing/data/cvpr20/work_space/')
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.val_list = '/home/users/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@3_dev_img_res_label_new.txt'
    conf.batch_size = 128
    conf.exp = '0213_res18_00_rgb_06_4@3'

    conf.model = edict()
    # conf.model.input = 'rgb' # choices=['three','rgb','depth','ir','nine_channel'], input modality for model
    conf.model.input_size = [336, 336] # [112,56] for half face, [56,56] for quarter face, [112,112] for whole face , first is width
    conf.model.embedding_size = 512
    conf.model.random_offset = [48,48] # first is for width, second is height

    conf.train = edict()
    conf.train.format = 'rgb'  # 'rgb', 'rgb_mix_nir', 'rgb_nir_pair','nir'
    conf.train.lr = 0.1
    conf.train.pretrained = False
    conf.train.transform = trans.Compose([
        # trans.Resize([conf.model.input_size[0] + 6, conf.model.input_size[1] + 6]),
        # trans.RandomCrop(conf.model.input_size),
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # trans.Normalize([0.5], [0.5])
    ])

    conf.eval = edict()
    conf.eval.format = 'rgb'
    conf.eval.transform = trans.Compose([
        # trans.Resize(conf.model.input_size),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

#--------------------Training Config ------------------------
    if training:        
        conf.train.momentum = 0.9
        conf.pin_memory = True
        conf.num_workers = 0
        conf.train.criterion_xent = torch.nn.SmoothL1Loss()
#--------------------Inference Config ------------------------
    else:
        conf.threshold = 1.5
        conf.face_limit = 10 
        conf.min_face_size = 30
    return conf