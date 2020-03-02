from data_pipe import get_train_loader,  get_train_loader_rgb, get_test_loader, get_test_loader_rgb, get_train_loader_rgb_crop, get_test_loader_rgb_crop
import torch
from torch import optim
# from matplotlib import pyplot as plt
# plt.switch_backend('agg')
from utils import AverageMeter, make_folder_if_not_exist
import os
import pprint
from resnet import resnet18, officali_resnet18, resnet18_concat
import torch.nn.functional as F
import numpy as np
from triplet_loss import triplet_semihard_loss
from config import get_config


# os.environ['CUDA_VISIBLE_DEVICES']='1'
conf = get_config()
gpu = conf.gpu

class face_learner(object):
    def __init__(self, conf, inference=False):
        pprint.pprint(conf)

        if conf.use_officical_resnet18:
            self.model = officali_resnet18()
            self.model = self.model.to(gpu)
        elif conf.use_concat:
            self.model = resnet18_concat(conf.model.use_senet, conf.model.embedding_size, conf.model.drop_out, conf.model.se_reduction, conf.use_triplet, conf.feature_c, conf.multi_output, conf.add)
            self.model = self.model.to(gpu)
        else:
            self.model = resnet18(conf.model.use_senet, conf.model.embedding_size, conf.model.drop_out, conf.model.se_reduction, conf.use_triplet, conf.rgb, conf.depth)
            # self.model = torch.nn.DataParallel(self.model).cuda()
            self.model = self.model.to(gpu)

        if not inference:
            if conf.rgb:
                print('We only use rgb images')
                if conf.crop:
                    self.loader = get_train_loader_rgb_crop(conf)
                else:
                    self.loader = get_train_loader_rgb(conf)
            else:
                self.loader = get_train_loader(conf)
                           
            self.optimizer = optim.SGD(list(self.model.parameters()), lr=conf.train.lr, momentum=conf.train.momentum)
            print(self.optimizer)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, conf.train.milestones, gamma=conf.train.gamma)
            print('optimizers generated')
            # self.print_freq = len(self.loader)//2
            self.print_freq = conf.print_freq
            print('print_freq: %d'%self.print_freq)
        else:
            if conf.rgb:
                if conf.crop:
                    self.test_loader = get_test_loader_rgb_crop(conf)
                else:
                    self.test_loader = get_test_loader_rgb(conf)
            else:
                self.test_loader = get_test_loader(conf)

    def save_state(self, save_path,  epoch):
        torch.save(self.model.state_dict(), save_path+'//'+'epoch={}.pth'.format(str(epoch)))

    def get_model_input_data(self, imgs):
        return torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).to(gpu)  # for rgb+depth+ir

    def get_model_input_data_rgb(self, imgs):
        # return imgs[0].cuda()  # for rgb
        return imgs[0].to(gpu)
    
    def get_model_input_data_rgb_for_test(self, imgs):
        # return imgs[0].cuda()  # for rgb
        return imgs[0].to(gpu)

    def get_model_input_data_for_test(self, imgs):
        input0 = torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).to(gpu)
        input1 = torch.cat((imgs[3], imgs[4], imgs[5]), dim=1).to(gpu)
        return input0, input1  # for rgb+depth+ir

    def eval_train(self, output, labels):
        output = F.softmax(output, dim=1)
        prob = torch.argmax(output, dim=1)
        train_acc = 0
        train_acc += (prob == labels).sum().item() / len(labels)
        return train_acc
    

    def train(self, conf):
        self.model.train()
        SL1_losses = AverageMeter()
        triplet_losses = AverageMeter()
        losses = AverageMeter()

        save_path = os.path.join(conf.save_path, conf.exp)
        make_folder_if_not_exist(save_path)

        for e in range(conf.train.epoches):
            
            print('exp {}'.format(conf.exp))
            print('epoch {} started'.format(e))
            print('learning rate: {}, {}'.format(len(self.scheduler.get_lr()), self.scheduler.get_lr()[0]))
            for batch_idx, (imgs, labels) in enumerate(self.loader):
                if conf.rgb:
                    input = self.get_model_input_data_rgb(imgs)
                else:
                    input = self.get_model_input_data(imgs)
                # print(input.shape)
                # labels = labels.cuda().float().unsqueeze(1)
                labels = labels.to(gpu)
                #print(labels)
                # print(labels.shape)
                # labels = torch.nn.functional.one_hot(labels, 2)
                if conf.use_triplet:
                    emd, output = self.model(input)
                else:
                    output = self.model(input)
               # print(emd)
                
                # output = self.model(input)
                # print(output.shape)
                # loss_SL1 = conf.train.criterion_SL1(output, labels)
                if conf.use_triplet:
                    if conf.use_label_smoothing:
                        loss_SL1 = conf.train.label_smoothing_loss(F.log_softmax(output, dim=1), labels)
                    else:
                        loss_SL1 = conf.train.softmax_loss(output, labels)
                    triplet_loss = triplet_semihard_loss(conf, labels, emd, conf.triplet_margin)
                    loss = loss_SL1 + triplet_loss * conf.triplet_ratio
                    train_acc = self.eval_train(output, labels)
                    triplet_losses.update(triplet_loss.item(), labels.size(0))
                    SL1_losses.update(loss_SL1.item(), labels.size(0))
                else:
                    if conf.use_label_smoothing:
                        loss_SL1 = conf.train.label_smoothing_loss(F.log_softmax(output, dim=1), labels)
                    else:
                        loss_SL1 = conf.train.softmax_loss(output, labels)
                   # loss_SL1 = conf.train.softmax_loss(output, labels)
                    train_acc = self.eval_train(output, labels)
                    loss = loss_SL1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), labels.size(0))
                SL1_losses.update(loss_SL1.item(), labels.size(0))

                
                if (batch_idx + 1) % self.print_freq == 0:
                    if conf.use_triplet:
                        print("Batch {}/{} Loss {:.6f} ({:.6f}) softmax_Loss {:.6f} ({:.6f}) triplet_Loss {:.6f} ({:.6f}) train_acc {:.6f}" \
                          .format(batch_idx + 1, len(self.loader), losses.val, losses.avg, SL1_losses.val, SL1_losses.avg, triplet_losses.val,
                                  triplet_losses.avg, train_acc))
                    else:
                        print("Batch {}/{} Loss {:.6f} ({:.6f}) SL1_Loss {:.6f} ({:.6f}) train_acc {:.6f}" \
                            .format(batch_idx + 1, len(self.loader), losses.val, losses.avg, SL1_losses.val,
                                    SL1_losses.avg, train_acc))
            self.scheduler.step()
            self.save_state(save_path, e)

    def test(self, conf):
        for epoch in range(conf.test.epoch_start, conf.test.epoch_end, conf.test.epoch_interval):
            save_listpath = os.path.join(conf.test.pred_path, conf.exp, conf.test.set, 'epoch={}.txt'.format(str(epoch)))
            make_folder_if_not_exist(os.path.dirname(save_listpath))
            fw = open(save_listpath, 'w')
            model_path = os.path.join(conf.save_path, conf.exp, 'epoch={}.pth'.format(str(epoch)))
            print(model_path)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            with torch.no_grad():
                for batch_idx, (imgs, names) in enumerate(self.test_loader):
                    if conf.rgb:
                        input = self.get_model_input_data_rgb_for_test(imgs)
                        if conf.use_triplet:
                            emd, output = self.model(input)
                        else:
                            output = self.model(input)
                        output = F.softmax(output, dim=1)
                        output = output[:,1]
                        #print(output.shape)
                        if conf.crop:
                            for k in range(len(names)):
                                save_img_path = '/'.join(names[k].split('/')[-4:])
                                write_str = save_img_path + ' %.6f' % output[k] + '\n'
                                fw.write(write_str)
                                fw.flush() 
                        else:
                            for k in range(len(names)):
                                write_str = names[k] + ' %.6f' % output[k] + '\n'
                                fw.write(write_str)
                                fw.flush()                         
                    else:
                        input1, input2 = self.get_model_input_data_for_test(imgs)
                        output1 = F.softmax(self.model(input1), dim=1)
                        output1 = output1[:,1]
                        output2 = F.softmax(self.model(input2), dim=1)
                        output2 = output2[:,1]
                        output = (output1 + output2 ) / 2.0
                        for k in range(len(names[0])):
                            write_str = names[0][k] + ' ' + '%.6f' % output[k] + '\n'
                            fw.write(write_str)
                            fw.flush()
            fw.close()

