import logging
import os
import pprint
import time

import numpy as np
import torch
from data import get_train_val_loader2
from resnet import resnet18
from torch import optim
from utils import make_folder_if_not_exist, mixup_criterion, mixup_data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class face_learner2(object):
    def __init__(self, conf, phase):
        self.config = conf
        self.logger = self.get_logger(phase)
        self.loader = get_train_val_loader2(conf)

        self.model = torch.nn.DataParallel(
            resnet18(conf.data.in_plane, **self.config.model.network)).cuda()

        if phase == 'Train':
            params = list(self.model.parameters())
            self.optimizer = optim.SGD(
                params,
                lr=conf.train.optimizer.lr,
                momentum=conf.train.optimizer.momentum,
                weight_decay=conf.train.optimizer.weight_decay)

            if conf.model.use_center_loss:
                self.optimizer_centloss = optim.SGD(
                    conf.train.criterion.cent.parameters(), lr=0.5)

            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                conf.train.optimizer.milestones,
                gamma=conf.train.optimizer.gamma)

    def get_logger(self, phase):
        log_format = '%(asctime)s - %(levelname)s: %(message)s'  # noqa
        log_path = os.path.join(
            'snapshots', self.config.job_name, 'log_{}.txt').format(phase)
        if log_path is not None:
            # mkdir
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
            pprint.pformat(self.config)))

        return logger

    def save_state(self, save_dir, epoch):
        save_name = 'model-%04d.pth' % (epoch)
        save_path = os.path.join(save_dir, save_name)
        self.logger.info('model saved: {}'.format(save_path))
        torch.save(self.model.state_dict(), save_path)

    def disp_batch(self, batch_idx, batch_loss, batch_acc, cent_loss=None):
        if cent_loss:
            self.logger.info('Train - Batch[{}]:\tLoss = {:.4f}\tAcc = {:.4f}\tCenterLoss = {:.4f}'.format(
                batch_idx + 1, batch_loss, batch_acc, cent_loss))
        else:
            self.logger.info('Train - Batch[{}]:\tLoss = {:.4f}\tAcc = {:.4f}'.format(
                batch_idx + 1, batch_loss, batch_acc))

    def disp_epoch(self, phase, running_loss, running_corrects, num_inst):
        epoch_loss = running_loss / num_inst
        epoch_acc = running_corrects.double() / num_inst
        self.logger.info(
            '{} - Batch All:\tLoss = {:.4f}\tAcc = {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    def run(self):
        self.logger.info('Exp: {}\n'.format(self.config.job_name))

        save_path = os.path.join(
            self.config.model.save_path, self.config.job_name)
        make_folder_if_not_exist(save_path)

        for e in range(self.config.train.epoches):
            since = time.time()
            self.logger.info(
                'Epoch {}/{}'.format(e, self.config.train.epoches-1))
            self.logger.info('-' * 20)
            self.logger.info('Learning rate: {}, {}'.format(
                len(self.scheduler.get_lr()), self.scheduler.get_lr()[0]))

            running_loss, running_corrects, num_inst = self.train(
                self.loader['Train'])
            self.disp_epoch('Train', running_loss, running_corrects, num_inst)

            self.save_state(save_path, e)
            self.scheduler.step()

            self.validation()
            time_elapsed = time.time() - since
            self.logger.info('time elapsed: {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))

    def train(self, loader):
        self.model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        num_inst = 0

        if self.config.model.use_center_loss:
            running_centloss = 0.

        for batch_idx, batch_samples in enumerate(loader):
            batch_data = batch_samples['image'].cuda()
            if self.config.model.use_relabel:
                batch_label = batch_samples['class'].cuda(
                ).float().unsqueeze(1)
            else:
                batch_label = batch_samples['class'].cuda()

            if self.config.model.use_mixup:
                # generate mixed inputs, two one-hot label vectors
                # and mixing coefficient
                batch_data, batch_label_A, batch_label_B, lam = \
                    mixup_data(batch_data, batch_label,
                               self.config.model.mixup_alpha, True)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            if self.config.model.use_center_loss:
                self.optimizer_centloss.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                features, outputs = self.model(batch_data)

                if self.config.model.use_relabel:
                    pred_labels = torch.where(outputs > 0.5, torch.ones(
                        1).cuda(), torch.zeros(1).cuda())
                else:
                    _, pred_labels = torch.max(outputs, 1)

                if self.config.model.use_mixup:
                    loss_func = mixup_criterion(
                        batch_label_A, batch_label_B, lam)
                    loss = loss_func(self.config.train.criterion.ce, outputs)
                else:
                    if self.config.model.use_relabel:
                        loss = self.config.train.criterion.sl1(
                            outputs, batch_label)
                    else:
                        loss = self.config.train.criterion.ce(
                            outputs, batch_label)

                if self.config.model.use_center_loss:
                    loss_cent = self.config.train.center_loss_weight * \
                        self.config.train.criterion.cent(features, batch_label)
                    loss += loss_cent

                loss.backward()
                self.optimizer.step()

                if self.config.model.use_center_loss:
                    for param in self.config.train.criterion.cent.parameters():
                        scale = 1. / self.config.train.center_loss_weight
                        param.grad.data *= scale
                    self.optimizer_centloss.step()

            # statistics
            running_loss += loss.item() * batch_label.size(0)
            if self.config.model.use_mixup:
                tmp_a = lam * \
                    pred_labels.eq(batch_label_A.data).cpu().sum()
                tmp_b = (1 - lam) * \
                    pred_labels.eq(batch_label_B.data).cpu().sum()
                running_corrects += (tmp_a + tmp_b)
            else:
                running_corrects += torch.sum(pred_labels ==
                                              batch_label.data)
            num_inst += batch_label.size(0)

            if self.config.model.use_center_loss:
                running_centloss += loss_cent.item() * batch_label.size(0)

            batch_loss = running_loss / num_inst
            batch_acc = running_corrects.double() / num_inst
            if ((batch_idx + 1) % 20) == 0:
                if self.config.model.use_center_loss:
                    cent_loss_ = running_centloss / num_inst
                    self.disp_batch(batch_idx, batch_loss,
                                    batch_acc, cent_loss_)
                else:
                    self.disp_batch(batch_idx, batch_loss, batch_acc)

        return running_loss, running_corrects, num_inst

    def validation(self):
        self.model.eval()   # Set model to evaluate mode

        for idx, val_loader in enumerate(self.loader['Val']):
            running_loss = 0.0
            running_corrects = 0
            num_inst = 0

            for batch_idx, batch_samples in enumerate(val_loader):
                batch_data = batch_samples['image'].cuda()
                if self.config.model.use_relabel:
                    batch_label = batch_samples['class'].cuda(
                    ).float().unsqueeze(1)
                else:
                    batch_label = batch_samples['class'].cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()
                if self.config.model.use_center_loss:
                    self.optimizer_centloss.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    features, outputs = self.model(batch_data)
                    if self.config.model.use_relabel:
                        pred_labels = torch.where(outputs > 0.5, torch.ones(
                            1).cuda(), torch.zeros(1).cuda())
                        loss = self.config.train.criterion.sl1(
                            outputs, batch_label)
                    else:
                        _, pred_labels = torch.max(outputs, 1)
                        loss = self.config.train.criterion.ce(
                            outputs, batch_label)

                    # statistics
                    running_loss += loss.item() * batch_label.size(0)
                    running_corrects += torch.sum(pred_labels ==
                                                  batch_label.data)
                    num_inst += batch_label.size(0)
            self.disp_epoch('Val#{}'.format(idx), running_loss,
                            running_corrects, num_inst)

    def test(self):
        for e in self.config.test.epochs:
            model_name = 'model-{:04d}.pth'.format(e)
            model_path = os.path.join(
                self.config.model.save_path, self.config.job_name, model_name)
            self.model.load_state_dict(torch.load(model_path))
            self.logger.info('Loading model {}...'.format(model_path))

            self.model.eval()

            collections = zip(self.config.test.save_name,
                              self.config.test_list, self.loader['Test'])
            for save_file_prefix, lst, loader in collections:
                save_file_name = '{}-epoch-{:02d}.txt'.format(
                    save_file_prefix, e)
                save_dir = os.path.join(
                    self.config.test.pred_path, self.config.job_name)

                make_folder_if_not_exist(save_dir)
                save_list_path = os.path.join(save_dir, save_file_name)
                self.logger.info('Processing: {}'.format(lst))
                self.logger.info('Writing to {}...'.format(save_list_path))

                lines = open(lst).readlines()

                with open(save_list_path, 'w') as fp:
                    since = time.time()

                    for batch_idx, batch_samples in enumerate(loader):
                        batch_data = batch_samples['image'].cuda()
                        batch_label = batch_samples['class'].cuda()

                        with torch.set_grad_enabled(False):
                            _, outputs = self.model(batch_data)
                            if self.config.model.use_relabel:
                                values = outputs.cpu().numpy().ravel()
                            else:
                                values = torch.softmax(
                                    outputs, 1).cpu().numpy()[:, 1]
                            for ind_, v in enumerate(values):
                                ind = ind_ + batch_idx * \
                                    self.config.model.batch_size
                                name, *_ = lines[ind].strip().split()
                                out_str = '{} {} {:.4f}\n'.format(
                                    name, batch_label[ind_], np.asscalar(v))
                                fp.write(out_str)

                    time_elapsed = time.time() - since
                    self.logger.info('Time elapsed: {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
