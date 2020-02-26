from data import get_train_val_loader,  get_test_loader
import torch
from torch import optim
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import AverageMeter, make_folder_if_not_exist
import os
import pprint
import numpy as np
from resnet import resnet18


os.environ['CUDA_VISIBLE_DEVICES']='1'

class face_learner(object):
    def __init__(self, conf, inference=False):
        pprint.pprint(conf)

        self.model = resnet18(conf.model.use_senet, conf.model.embedding_size, conf.model.drop_out, conf.model.se_reduction)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.loader = get_train_val_loader(conf)

        if not inference:
            self.optimizer = optim.SGD(list(self.model.parameters()), lr=conf.train.lr, momentum=conf.train.momentum)
            print(self.optimizer)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, conf.train.milestones, gamma=conf.train.gamma)
            print('Optimizers generated...')
            self.print_freq = len(self.loader) // 2
            print('Display interval = %d' % self.print_freq)
        # else:
        #     self.test_loader = get_test_loader(conf)

    def save_state(self, save_path,  epoch):
        torch.save(self.model.state_dict(), save_path+'//'+'epoch={}.pth'.format(str(epoch)))

    def get_model_input_data(self, imgs):
        print(imgs)
        return torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).cuda()  # for rgb+depth+ir

    def get_model_input_data_for_test(self, imgs):
        input0 = torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).cuda()
        input1 = torch.cat((imgs[3], imgs[4], imgs[5]), dim=1).cuda()
        return input0, input1  # for rgb+depth+ir

    def train(self, conf):
        print('\nExp: {}\n'.format(conf.exp))

        save_path = os.path.join(conf.save_path, conf.exp)
        make_folder_if_not_exist(save_path)

        for e in range(conf.train.epoches):
            print('\nEpoch {}/{}'.format(e, conf.train.epoches-1))
            print('-' * 20)
            print('Learning rate: {}, {}'.format(len(self.scheduler.get_lr()), self.scheduler.get_lr()[0]))

            # Each epoch has a training and validation phase
            for phase in ['Train', 'Val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                num_inst = 0

                for batch_idx, batch_samples in enumerate(self.loader[phase]):
                    batch_data = batch_samples['image'].cuda()
                    batch_label = batch_samples['class'].cuda()

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):
                        outputs = self.model(batch_data)
                        _, preds = torch.max(outputs, 1)
                        loss = conf.train.criterion_ce(outputs, batch_label)

                        if phase == 'Train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_label.size(0)
                    running_corrects += torch.sum(preds == batch_label.data)
                    num_inst += batch_label.size(0)

                    if phase == 'Train':
                        batch_loss = running_loss / num_inst
                        batch_acc = running_corrects.double() / num_inst
                        if ((batch_idx + 1) % 20) == 0:
                            print('{} - Batch[{}]:\tLoss = {:.4f}\tAcc = {:.4f}'.format(phase, batch_idx + 1, batch_loss, batch_acc))
                if phase == 'Train':
                    self.save_state(save_path, e)
                    self.scheduler.step()

                epoch_loss = running_loss / num_inst
                epoch_acc = running_corrects.double() / num_inst
                print('{} - Batch All:\tLoss = {:.4f}\tAcc = {:.4f}'.format(phase, epoch_loss, epoch_acc))


    def test(self, conf):
        save_listpath = os.path.join(conf.test.pred_path, conf.exp, '{}-epoch-{}.txt'.format(conf.test.save_name, conf.test.epoch))
        make_folder_if_not_exist(os.path.dirname(save_listpath))
        model_path = os.path.join(conf.save_path, conf.exp, 'epoch={}.pth'.format(conf.test.epoch))

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        lines = open(conf.test_list).readlines()

        with open(save_listpath, 'w') as fp:
            with torch.no_grad():
                for batch_idx, batch_samples in enumerate(self.loader['Test']):
                    batch_data = batch_samples['image'].cuda()
                    batch_label = batch_samples['class'].cuda()

                    with torch.set_grad_enabled(False):
                        outputs = self.model(batch_data)
                        values = torch.softmax(outputs, 1).cpu().numpy()[:, 1]
                        for ind_, v in enumerate(values):
                            ind = batch_idx * conf.batch_size + ind_
                            name, *_ = lines[ind].strip().split()
                            out_str = '{} {}\n'.format(name, np.asscalar(v.round(4)))
                            fp.write(out_str)

