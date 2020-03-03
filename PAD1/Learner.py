from data_pipe import get_train_loader, get_val_loader, get_test_loader
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from torchvision import transforms as trans
import os
import shutil
import pandas as pd
from resnet_new import resnet_face50, resnet_face101
from easydict import EasyDict as edict
import time
import logging
import pprint
from utils import AverageMeter



os.environ['CUDA_VISIBLE_DEVICES']='3'

class face_learner(object):
    def __init__(self, conf, inference=False):
        self.conf = conf
        self.logger = self.get_logger()

        if self.conf.model.format == 'res50':
            self.model = resnet_face50(use_se=True)
        elif self.conf.model.format == 'res101':
            self.model = resnet_face101(use_se=True)
        else:
            raise ValueError
        self.model = torch.nn.DataParallel(self.model).cuda()

        if not inference:
            self.milestones = conf.train.milestones
            self.loader = get_train_loader(conf)
            self.step = 0
            self.optimizer = optim.SGD(list(self.model.parameters()), lr=conf.train.lr,
                                           momentum=conf.train.momentum, weight_decay=0.0005)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, gamma=conf.train.gamma)
            self.print_freq = len(self.loader)//10
            self.val_loader = get_val_loader(conf)

        else:
            self.model.load_state_dict(torch.load(conf.model_path))

    def get_logger(self):
        log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        log_file = '{}_{}.log'.format(self.conf.exp,
                                         time.strftime('%Y-%m-%d-%H-%M'))
        log_path = os.path.join(str(self.conf.log_path), log_file)
        if log_path is not None:
            if os.path.exists(log_path):
                os.remove(log_path)
            log_dir = os.path.dirname(log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # get logger
            logger = logging.getLogger()
            logger.handlers = []
            formatter = logging.Formatter(log_format)
            # file handler
            handler = logging.FileHandler(log_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # stream handler
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # set level (info)
            logger.setLevel(logging.INFO)
            logging.basicConfig(level=logging.INFO, format=log_format)
        else:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logging.basicConfig(level=logging.INFO, format=log_format)
        logger.info('\n############# Config #############\n{}'.format(
            pprint.pformat(self.conf)))
        return logger

    def save_state(self, save_path,  epoch, model_only=True):
        if self.conf.train.format == 'rgb':
            torch.save(self.model.state_dict(), save_path+'//'+'epoch={}.pth'.format(str(epoch)))
        elif self.conf.train.format == 'rgb_mix_nir':
            torch.save(self.model.state_dict(), save_path+'//'+'epoch={}.pth'.format(str(epoch)))
        elif self.conf.train.format == 'nir':
            torch.save(self.model.state_dict(), save_path + '//' + 'epoch={}.pth'.format(str(epoch)))
        elif self.conf.train.format == 'depth':
            torch.save(self.model.state_dict(), save_path + '//' + 'epoch={}.pth'.format(str(epoch)))
        elif self.conf.train.format == 'rgb_nir_pair':
            torch.save(self.model.state_dict(), save_path+'//'+'rgb_epoch={}.pth'.format(str(epoch)))
            torch.save(self.model1.state_dict(), save_path+'//'+'nir_epoch={}.pth'.format(str(epoch)))
        if not model_only:
            torch.save(self.optimizer.state_dict(), save_path+'//'+'optimizer_{}_step={}.pth'.format(get_time(), self.step))

    def make_dirfolder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


    def test_reg(self, conf):
        test_loader = get_test_loader(conf)
        result_path = os.path.join(conf.result_path, conf.exp, conf.result_name)
        self.make_dirfolder(os.path.dirname(result_path))
        fw = open(result_path, 'w')
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, names) in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print('processing %d batch ...'%batch_idx)
                input = self.get_model_input_data(imgs, conf.eval.format)
                output, feat = self.model(input[0])
                output.squeeze_(1)
                for k in range(len(names[0])):
                    # write_str = names[0][k]+' '+names[1][k]+' '+names[2][k]+' '+'%.10f'%output[k]+'\n'
                    write_str = names[0][k]+' '+'%.10f'%output[k]+'\n'
                    fw.write(write_str)
        fw.close()
        print('Testing Completed!')

    def val_reg(self, conf):
        test_loader = get_val_loader(conf)
        result_path = os.path.join(conf.result_path, conf.exp, conf.result_name)
        self.make_dirfolder(os.path.dirname(result_path))
        fw = open(result_path, 'w')
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, labels, names) in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print('processing %d batch ...'%batch_idx)
                input = self.get_model_input_data(imgs, conf.eval.format)
                output, feat = self.model(input[0])
                output.squeeze_(1)
                for k in range(len(names[0])):
                    write_str = names[0][k]+' '+'%f'%labels[k]+' '+'%.10f'%output[k]+'\n'
                    fw.write(write_str)
        fw.close()
        print('Testing Completed!')

    def get_model_input_data(self, imgs, format):
        if format == 'rgb_mix_nir':
            imgs[0] = imgs[0].cuda()
            input = [imgs[0]]
        elif format == 'rgb':
            imgs[0] = imgs[0].cuda()
            input = [imgs[0]]
        elif format == 'nir':
            imgs[0] = imgs[0].cuda()
            input = [imgs[0]]
        elif format == 'depth':
            imgs[0] = imgs[0].cuda()
            input = [imgs[0]]
        elif format == 'rgb_nir_pair':
            imgs[0] = imgs[0].cuda()
            imgs[1] = imgs[1].cuda()
            input = [imgs[0], imgs[1]]
        else:
            assert False, 'only support conf.train.format = \'rgb\', \'rgb_mix_nir\', \'rgb_nir_pair\''
        return input

    def train(self, conf, epochs):
        self.model.train()
        if self.conf.train.format == 'rgb_nir_pair':
            self.model1.train()
        xent_losses = AverageMeter()
        xent1_losses = AverageMeter()
        xent2_losses = AverageMeter()
        ment_losses = AverageMeter()
        losses = AverageMeter()

        save_path = str(conf.save_path/conf.exp)
        self.make_dirfolder(save_path)

        for e in range(epochs):
            print('exp {}'.format(conf.exp))
            batch_tic = time.time()
            for batch_idx, (imgs, labels, _) in enumerate(self.loader):
                input = self.get_model_input_data(imgs, conf.train.format)
                labels = labels.cuda().float().unsqueeze(1)
                output, feat = self.model(input[0])
                loss_xent = conf.train.criterion_xent(output, labels)
                if self.conf.train.format == 'rgb_nir_pair':
                    output1, feat1, b = self.model1([input[1], conf.model.use_senet])
                    loss_xent1 = conf.train.criterion_xent1(output1, labels)
                    loss_xent2 = conf.train.criterion_xent2(feat, feat1).mean()
                if self.conf.train.format == 'rgb_nir_pair':
                    loss = loss_xent + loss_xent1 + loss_xent2
                else:
                    loss = loss_xent

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), labels.size(0))
                xent_losses.update(loss_xent.item(), labels.size(0))
                acc_cur = 0 # for regression
                if self.conf.train.format == 'rgb_nir_pair':
                    xent1_losses.update(loss_xent1.item(), labels.size(0))
                    xent2_losses.update(loss_xent2.item(), labels.size(0))
                    predictions1 = output1.data.max(1)[1]
                    correct1 = (predictions1 == labels.data).sum()
                    acc_cur1 = correct1 * 100. / labels.size(0)
                else:
                    acc_cur1 = 0
                if (batch_idx + 1) % self.print_freq == 0:
                    speed = self.conf.batch_size * \
                            (batch_idx + 1) / (time.time() - batch_tic)
                    s = 'Epoch[{}] Batch[{}] Speed: {:.2f} samples/sec lr: {:.8f}'.format(
                            e+1, batch_idx, speed, self.scheduler.get_lr()[0])
                    s += ' loss {:.6f} ({:.6f}) acc {:.2f} '.format(losses.val, losses.avg, acc_cur)
                    self.logger.info(s)

                self.step += 1
            self.save_state(save_path, e)
            self.scheduler.step()
        logging.info('Training Completed!')