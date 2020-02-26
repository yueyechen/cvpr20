from easydict import EasyDict as edict
from torchvision import transforms as trans
import torch.nn as nn

def get_config(training = True):
    conf = edict()
    conf.model = edict()
    conf.train = edict()
    conf.eval = edict()

    conf.data_folder = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1' #data root for training, and testing, you should change is according to your setting
    conf.save_path = './snapshots' #path for save model in training process, you should change it according to your setting
    conf.train_list =  '4@3_train.txt' #path where training list is, you should change it according to your setting
    conf.val_list =  '4@3_dev_val_res_label.txt' #path where validation list is, you should change it according to your setting
    conf.test_list = '4@3_dev_test_res_label.txt' #path where test list is, you should change it according to your setting
    conf.batch_size = 128
    conf.exp = 'test-20200208-4@3' #model is saved in conf.save_path/conf.exp, if you want to train different models, you can distinguish them according to this parameter

    conf.model.crop_size = [128,128] #the crop size of our model
    conf.model.input_size = [112,112] #the input size of our model
    conf.model.expand_ratio = 1.2 #the expand_ratio of bbox
    conf.model.random_offset = [16,16] #for random crop
    conf.model.use_senet = True #senet is adopted in our resnet18 model
    conf.model.se_reduction = 16 #parameter concerning senet
    conf.model.drop_out = 0.7 #we add dropout layer in our resnet18 model
    conf.model.embedding_size = 1024 #feature size of our resnet18 model

    conf.pin_memory = True
    conf.num_workers = 3

#--------------------Training Config ------------------------
    # if training:
    conf.train.lr = 0.01 # the initial learning rate
    conf.train.milestones = [80, 140, 180] #epoch milestones decreased by a factor of 10
    conf.train.epoches = 10 #we trained our model for 200 epoches
    conf.train.momentum = 0.9 #parameter in setting SGD
    conf.train.gamma = 0.1 #parameter in setting lr_scheduler

    conf.train.criterion_SL1 = nn.SmoothL1Loss() #we use SmoothL1Loss in training stage
    conf.train.criterion_ce = nn.CrossEntropyLoss() #we use CE in training stage

    conf.train.transform = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
        # trans.RandomCrop(config.model.input_size)
        trans.CenterCrop(conf.model.input_size),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

#--------------------Inference Config ------------------------
    conf.test = edict()
    conf.test.save_name = '4@3-dev-res'
    conf.test.epoch = 9
    # conf.test.epoch_start = 150
    # conf.test.epoch_end = 200
    # conf.test.epoch_interval = 8 #we set a range of epoches for testing
    conf.test.pred_path = './dump' #path for save predict result, pred_result is saved in conf.pred_path/conf.exp, you should change it according to your setting
    conf.test.transform = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
        trans.CenterCrop(conf.model.input_size),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return conf
