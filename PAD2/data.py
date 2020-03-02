# coding=utf-8

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_train_val_loader(conf):
    print('\nTrain dataset: {}'.format(conf.train_list))
    print('Val dataset: {}'.format(conf.val_list))
    print('Test dataset: {}\n'.format(conf.test_list))

    datasets = {}
    for phase in ['Train', 'Val', 'Test']:
        datasets[phase] = SMDataset(conf, phase=phase)

    dataloader = {}
    for phase in ['Train', 'Val', 'Test']:
        if phase == 'Train':
            dataloader[phase] = DataLoader(datasets[phase], batch_size=conf.batch_size,
                                           shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
        else:
            dataloader[phase] = DataLoader(datasets[phase], batch_size=conf.batch_size,
                                           shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)

    return {x: dataloader[x] for x in ['Train', 'Val', 'Test']}


def get_train_val_loader2(conf):
    print('\nTrain dataset: {}'.format(conf.train_list))
    print('Val dataset: {}'.format(conf.val_list))
    print('Test dataset: {}\n'.format(conf.test_list))

    datasets = {}
    datasets['Train'] = SMDataset2(
        lst_file=conf.train_list, phase='Train', config=conf.data)

    datasets['Val'] = []
    for lst in conf.val_list:
        datasets['Val'].append(SMDataset2(
            lst_file=lst, phase='Val', config=conf.data))

    datasets['Test'] = []
    for lst in conf.test_list:
        datasets['Test'].append(SMDataset2(
            lst_file=lst, phase='Test', config=conf.data))

    dataloader = {}
    dataloader['Train'] = DataLoader(datasets['Train'],
                                     batch_size=conf.model.batch_size,
                                     shuffle=True,
                                     pin_memory=conf.data.pin_memory,
                                     num_workers=conf.data.num_workers)

    dataloader['Val'] = []
    for ds in datasets['Val']:
        dataloader['Val'].append(
            DataLoader(ds,
                       batch_size=conf.model.batch_size,
                       shuffle=False,
                       pin_memory=conf.data.pin_memory,
                       num_workers=conf.data.num_workers))

    dataloader['Test'] = []
    for ds in datasets['Test']:
        dataloader['Test'].append(
            DataLoader(ds,
                       batch_size=conf.model.batch_size,
                       shuffle=False,
                       pin_memory=conf.data.pin_memory,
                       num_workers=conf.data.num_workers))

    return {x: dataloader[x] for x in ['Train', 'Val', 'Test']}


def load_image(path, itype):
    img = Image.open(path).convert(itype)
    return img


def norm_roi(rects, im_shape, phase, crop_shape=(128, 128), input_shape=(112, 112), expand_ratio=1.0):
    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    im_h, im_w = im_shape
    if h > w:
        origin = rects[0] + rects[2]
        rects[0] = max((origin / 2. - h / 2.), 0)
        rects[2] = min((origin / 2. + h / 2.), im_w)
    else:
        origin = rects[1] + rects[3]
        rects[1] = max((origin / 2. - w / 2.), 0)
        rects[3] = min((origin / 2. + w / 2.), im_h)

    w = rects[2] - rects[0]
    h = rects[3] - rects[1]
    bbox1 = [max(rects[0] - w * (expand_ratio - 1) / 2., 0),
             max(rects[1] - h * (expand_ratio - 1) / 2., 0),
             min(rects[2] + w * (expand_ratio - 1) / 2., im_w),
             min(rects[3] + h * (expand_ratio - 1) / 2., im_h)]
    bbox1 = [int(x) for x in bbox1]

    if phase in ['Val', 'Test']:
        return bbox1

    crop_w = bbox1[2] - bbox1[0]
    crop_h = bbox1[3] - bbox1[1]
    ratio = crop_shape[0] / input_shape[0]
    bbox2 = [max(bbox1[0] - crop_w * (ratio - 1) / 2., 0),
             max(rects[1] - crop_h * (ratio - 1) / 2., 0),
             min(rects[2] + crop_w * (ratio - 1) / 2., im_w),
             min(rects[3] + crop_h * (ratio - 1) / 2., im_h)]
    bbox2 = [int(x) for x in bbox2]

    return bbox2


class SMDataset2(Dataset):
    def __init__(self, lst_file, phase, config):
        self.phase = phase
        self.transform = config.train_transform if phase == 'Train' \
            else config.test_transform

        self.root = config.folder
        self.crop_size = config.crop_size
        self.input_size = config.input_size
        self.expand_ratio = config.expand_ratio
        self.use_multi_color = config.use_multi_color
        self.in_data_format = config.in_data_format

        self.info = []
        self.lines = open(lst_file).readlines()
        for l in self.lines:
            path, *roi, label = l.strip().split()
            self.info.append((path, roi, float(label)))

        mean_ = [0.5] * config.in_plane
        std_ = [0.5] * config.in_plane
        self.im_norm = transforms.Normalize(mean_, std_)

    def preprocess_image(self, roi_raw, data_in):
        if False:
            im_face = im.resize(self.crop_size)
        else:
            if self.use_multi_color:
                if roi_raw == [-1] * 4:
                    data_out = [x.resize(self.crop_size) for x in data_in]
                else:
                    roi_out = norm_roi(roi_raw, data_in[0].size, self.phase,
                                       self.crop_size, self.input_size,
                                       self.expand_ratio)
                    data_out = [x.crop(roi_out).resize(
                        self.crop_size) for x in data_in]
            else:
                if roi_raw == [-1] * 4:
                    data_out = data_in.resize(self.crop_size)
                else:
                    roi_out = norm_roi(roi_raw, data_in.size, self.phase,
                                       self.crop_size, self.input_size,
                                       self.expand_ratio)
                    data_out = data_in.crop(roi_out).resize(self.crop_size)

            return data_out

    def read_image(self, index):
        path, sroi, label = self.info[index]
        roi_raw = [int(float(x)) for x in sroi]

        if self.use_multi_color:
            ims = [load_image(os.path.join(self.root, path), color)
                   for color in self.in_data_format]
            data_out = self.preprocess_image(roi_raw, ims)

            if self.transform:
                data_trans = [self.transform(x) for x in data_out]
            im_out = torch.cat(data_trans, dim=0)
        else:
            im = load_image(os.path.join(
                self.root, path), self.in_data_format)
            data_out = self.preprocess_image(roi_raw, im)
            if self.transform:
                im_out = self.transform(data_out)

        return self.im_norm(im_out), label

    def __getitem__(self, index):
        im_out, label = self.read_image(index)

        return {'image': im_out, 'class': label}

    def __len__(self):
        return len(self.info)


if __name__ == "__main__":
    import importlib
    import sys

    def _func(loader):
        for i_batch, sample_batched in enumerate(loader):
            # print(i_batch, sample_batched['image'].size(),
            #       sample_batched['class'].size())
            for i, x in enumerate(sample_batched['image']):
                save_name = '%06d.jpg' % (i)
                save_path = os.path.join(save_dir, save_name)
                t(x * 0.5 + 0.5).save(save_path)
            break
    conf_file = sys.argv[1]
    conf = importlib.import_module(conf_file.strip(
        '.py').replace('/', '.')).get_config()

    loader = get_train_val_loader2(conf)
    phase = 'Train'
    t = transforms.ToPILImage()

    save_dir = 'dump'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if phase == 'Train':
        _func(loader['Train'])
    else:
        for idx, ldr in enumerate(loader[phase]):
            print('\n#{} Loader No.{}'.format(phase, idx+1))
            _func(ldr)
