from easydict import EasyDict as edict
from torchvision import transforms as trans
import torch.nn as nn
from label_smoothing import LabelSmoothing, LabelSmoothingLoss

def get_config(training = True):
    conf = edict()
    conf.model = edict()
    conf.train = edict()
    conf.eval = edict()


    conf.gpu = 3
    conf.use_concat = False
    conf.multi_output = True
    conf.add = True
    conf.feature_c = 128
    conf.use_triplet = False
    conf.use_officical_resnet18 = False
    conf.triplet_ratio = 0.01
    conf.triplet_margin = 0.2
    conf.print_freq = 20
    conf.rgb = True
    conf.depth = False
    conf.ir = True
    conf.crop = True
    conf.use_label_smoothing = True

    conf.data_folder = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase2' #data root for training, and testing, you should change is according to your setting
    conf.save_path = '/mnt/cephfs/smartauto/users/guoli.wang/tao.cai/cvpr_model'
    #conf.save_path = './work_space/save' #path for save model in training process, you should change it according to your setting
    conf.train_list =  '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@2_train.txt' #path where training list is, you should change it according to your setting
   # conf.train_list = '/tmp/yuxi.feng/4@2.det'
   # conf.train_list = '/tmp/yuxi.feng/IR/4@2_train.txt'
    conf.test_list = '/tmp/yuxi.feng/IR/4@2_test.txt'
   # conf.test_list = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase2/4@2_test_res.txt'
   # conf.test_list = '/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1/4@2_dev_res.txt' #path where test list is, you should change it according to your setting
   # conf.test_list = '/tmp/yuxi.feng/4@1_test.det'
    conf.batch_size = 128

   # conf.exp = 'phase1_ir_4@2_lr_0.001_epoch_30_input_size_256_emd_128_dropout_0.0_triplet_ratio_0.01_wo_se_offset_32' 
   # conf.exp = 'phase1_ir_4@2_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_offset_32_smooth_fix_ir'
    conf.exp = 'phase1_ir_4@2_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_crop_offset_32_smooth'
   # conf.exp = 'phase1_depth_4@1_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_offset_32_smooth'
   # conf.exp = 'phase1_multi_4@3_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_offset_32'
   # conf.exp = 'phase1_4@2_lr_0.001_epoch_30_input_size_256_emd_128_dropout_0.0_triplet_ratio_0.01_fix_wo_se_crop_offset_32' #model is saved in conf.save_path/conf.exp, if you want to train different models, you can distinguish them according to this parameter
   # conf.exp = 'phase1_4@2_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_offset_32'
   # conf.exp = 'phase1_4@3_lr_0.01_epoch_30_input_size_384_emd_128_dropout_0.0_wo_se_crop_offset_32_smooth'
   # conf.exp = 'phase1_4@1_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_crop_offset_32_smooth_twice' # origin smooth
   # conf.exp = 'phase1_4@2_lr_0.1_epoch_30_input_size_256_emd_128_dropout_0.0_se_crop_offset_32_smooth'  # with se
   # conf.exp = 'phase1_4@2_lr_0.1_epoch_30_input_size_256_emd_128_dropout_0.0_wo_se_wo_crop_offset_32_smooth'  # without crop
   # conf.exp = 'phase1_4@3_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_triplet_raito_0.01_wo_se_crop_offset_32_smooth'  # add triplet loss
   # conf.exp = 'phase1_4@2_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_triplet_ratio_0.01_wo_se_offset_32'
   # conf.exp = 'phase1_4@3_lr_0.01_epoch_30_input_size_256_emd_128_dropout_0.0_triplet_ratio_0.01_fix_crop_offset_32'
   # conf.exp = 'phase1_4@3_lr_0.001_epoch_30_input_size_384_emd_128_dp_0_concat_128_bs_128_crop' #model is saved in conf.save_path/conf.exp, if you want to train different models, you can distinguish them according to this parameter
   # conf.exp = 'phase1_4@1_resnet18_lr_0.01_epoch_30_input_size_256_emd_512_bs_128_offset_32_fix'
   # conf.exp = 'phase1_4@1_lr_0.001_epoch_30_input_size_512_emd_128_bs_64'
    #conf.exp = 'phase1_4@1_resnet18_lr_0.001_epoch_30_input_size_512_emd_128_bs_64_test'
   # conf.exp = 'phase1_4@2_lr_0.001_epoch_30_input_size_320_emd_128_dp_0_concat_128_bs_128'
   # conf.exp = 'phase1_4@3_lr_0.001_epoch_30_input_size_384_emd_128_dp_0_concat_128_bs_128_fix_bug'
   # conf.exp = 'phase1_4@1_lr_0.001_epoch_30_input_size_384_emd_128_dp_0_wo_mo_concat_128_bs_64'
   # conf.exp = 'phase1_4@1_lr_0.001_epoch_30_input_size_384_emd_128_dp_0_wo_mo_add_128_bs_128_fix'

    conf.model.input_size = 256 #the input size of our model
    conf.model.random_offset = 32 #for random crop
    conf.model.use_senet = False #senet is adopted in our resnet18 model
    conf.model.se_reduction = 16 #parameter concerning senet
    conf.model.drop_out = 0.0 #we add dropout layer in our resnet18 model
    conf.model.embedding_size = 128 #feature size of our resnet18 model

    conf.pin_memory = True
    conf.num_workers = 3

#--------------------Training Config ------------------------
    if training:
        conf.train.lr = 0.01 # the initial learning rate
        conf.train.milestones = [10, 20, 25] #epoch milestones decreased by a factor of 10
        conf.train.epoches = 30 #we trained our model for 200 epoches
        conf.train.momentum = 0.9 #parameter in setting SGD
        conf.train.gamma = 0.1 #parameter in setting lr_scheduler
        conf.train.criterion_SL1 = nn.SmoothL1Loss() #we use SmoothL1Loss in training stage
        conf.train.softmax_loss = nn.CrossEntropyLoss() # we use cross-entropyloss for rgb classification
        conf.train.label_smoothing_loss = LabelSmoothing(size=2, padding_idx=0, smoothing=0.1) # for label smoothing
        conf.train.label_smoothing_loss1 = LabelSmoothingLoss(0.1, 2, None)

        conf.train.transform = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
            trans.Resize((conf.model.input_size, conf.model.input_size)),
            trans.RandomCrop((conf.model.input_size - conf.model.random_offset, conf.model.input_size - conf.model.random_offset)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        conf.train.transform1 = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
            trans.Resize((conf.model.input_size, conf.model.input_size)),
            trans.RandomCrop((conf.model.input_size - conf.model.random_offset, conf.model.input_size - conf.model.random_offset)),
            trans.ToTensor(),
            trans.Normalize([0.5,], [0.5,])
        ])
        conf.train.transform2 = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
            trans.Resize((conf.model.input_size, conf.model.input_size)),
            trans.RandomCrop((conf.model.input_size - conf.model.random_offset, conf.model.input_size - conf.model.random_offset)),
            trans.ToTensor(),
            trans.Normalize([0.5,], [0.5,])
        ])

#--------------------Inference Config ------------------------
    else:
        conf.test = edict()
        conf.test.set = 'phase2_test'
       # conf.test.set = '4@2_test'
        conf.test.epoch_start = 9
        conf.test.epoch_end = 30
        conf.test.epoch_interval = 10 #we set a range of epoches for testing
        #conf.test.pred_path = '/home/users/tao.cai/PAD/work_space/test_pred' #path for save predict result, pred_result is saved in conf.pred_path/conf.exp, you should change it according to your setting
        conf.test.pred_path = '/mnt/cephfs/smartauto/users/guoli.wang/tao.cai/cvpr_results'
        conf.test.transform = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
           # trans.Resize((conf.model.input_size,conf.model.input_size)),
            trans.Resize((conf.model.input_size - conf.model.random_offset, conf.model.input_size - conf.model.random_offset)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        conf.test.transform1 = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
           # trans.Resize((conf.model.input_size,conf.model.input_size)),
            trans.Resize((conf.model.input_size - conf.model.random_offset, conf.model.input_size - conf.model.random_offset)),
            trans.ToTensor(),
            trans.Normalize([0.5,], [0.5,])
        ])
        conf.test.transform2 = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
           # trans.Resize((conf.model.input_size,conf.model.input_size)),
            trans.Resize((conf.model.input_size - conf.model.random_offset, conf.model.input_size - conf.model.random_offset)),
            trans.ToTensor(),
            trans.Normalize([0.5,], [0.5,])
        ])
        

    return conf
