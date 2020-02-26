# coding=utf-8

import os
import random

from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
            dataloader[phase] = DataLoader(datasets[phase], batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
        else:
            dataloader[phase] = DataLoader(datasets[phase], batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)

    return {x: dataloader[x] for x in ['Train', 'Val', 'Test']}


def get_test_loader(conf):
    print('test dataset: {}'.format(conf.test_list))

    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)

    return {'Test': loader}


def default_loader_rgb(path):
    img = Image.open(path).convert('RGB')
    return img


def default_loader_gray(path):
    img = Image.open(path).convert('L')
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


class SMDataset(Dataset):
    def __init__(self, conf, phase, target_transform=None, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        self.phase = phase
        self.transform = conf.test.transform if phase == 'Test' else conf.train.transform
        self.target_transform = target_transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder

        self.crop_size = conf.model.crop_size
        self.input_size = conf.model.input_size
        self.expand_ratio = conf.model.expand_ratio

        # self.random_offset = conf.model.random_offset

        if phase == 'Train':
            self.lines = open(conf.train_list).readlines()
        elif phase == 'Val':
            self.lines = open(conf.val_list).readlines()
        else:
            self.lines = open(conf.test_list).readlines()

        self.info = []
        for l in self.lines:
            path, *roi, label = l.strip().split()
            self.info.append((path, roi, int(label)))

    def __getitem__(self, index):
        path, sroi, label = self.info[index]
        im = self.loader_rgb(os.path.join(self.root, path))
        roi_raw = [int(float(x)) for x in sroi]
        if roi_raw == [-1] * 4:
            im_face = im.resize(self.crop_size)
        else:
            roi_out = norm_roi(roi_raw, im.size, self.phase, self.crop_size, self.input_size, self.expand_ratio)
            im_face = im.crop(roi_out).resize(self.crop_size)

        if self.transform:
            im_resize = self.transform(im_face)

        return {'image': im_resize, 'class': label}

    def __len__(self):
        return len(self.info)


class MyDataset_huoti_test(Dataset):
    def __init__(self, conf, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        fh = open(conf.test_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1], words[2]))

        self.imgs = imgs
        self.transform = conf.test.transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder

    def __getitem__(self, index):
        fn1, fn2, fn3= self.imgs[index]
        img11 = self.loader_rgb(os.path.join(self.root,fn1))
        img12 = self.loader_gray(os.path.join(self.root,fn2))
        img13 = self.loader_gray(os.path.join(self.root,fn3))
        size_c = (8, 8, 120, 120)
        img11 = img11.crop(size_c)
        img12 = img12.crop(size_c)
        img13 = img13.crop(size_c)

        img21 = self.loader_rgb(os.path.join(self.root, fn1)).transpose(Image.FLIP_LEFT_RIGHT)
        img22 = self.loader_gray(os.path.join(self.root, fn2)).transpose(Image.FLIP_LEFT_RIGHT)
        img23 = self.loader_gray(os.path.join(self.root, fn3)).transpose(Image.FLIP_LEFT_RIGHT)
        img21 = img21.crop(size_c)
        img22 = img22.crop(size_c)
        img23 = img23.crop(size_c)

        if self.transform is not None:
            img11 = self.transform(img11)
            img12 = self.transform(img12)
            img13 = self.transform(img13)
            img21 = self.transform(img21)
            img22 = self.transform(img22)
            img23 = self.transform(img23)
        return [img11,img12,img13,img21,img22,img23], [fn1, fn2, fn3]

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    from config import get_config

    conf = get_config(training=True)
    loader = get_train_val_loader(conf)
    phase = 'Train'

    for i_batch, sample_batched in enumerate(loader[phase]):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['class'].size())

