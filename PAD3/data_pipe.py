# coding=utf-8
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import csv
import cv2
import numpy as np
import copy

def get_train_loader(conf):
    print('train dataset: {}'.format(conf.train_list))
    ds = MyDataset_huoti(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def get_train_loader_rgb(conf):
    print('train dataset: {}'.format(conf.train_list))
    ds = MyDataset_huoti_rgb(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def get_train_loader_rgb_crop(conf):
    print('train dataset: {}'.format(conf.train_list))
    ds = MyDataset_huoti_rgb_crop(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def get_test_loader(conf):
    print('test dataset: {}'.format(conf.test_list))
    ds = MyDataset_huoti_test(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def get_test_loader_rgb(conf):
    print('test dataset: {}'.format(conf.test_list))
    ds = MyDataset_huoti_test_rgb(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def get_test_loader_rgb_crop(conf):
    print('test dataset: {}'.format(conf.test_list))
    ds = MyDataset_huoti_rgb_test_crop(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader
    

def default_loader_rgb(path):
    img = Image.open(path).convert('RGB')
    return img

def default_loader_gray(path):
    img = Image.open(path).convert('L')
    return img

class MyDataset_huoti(Dataset):
    def __init__(self, conf, target_transform=None, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            rgb_img_path = words[0]
            depth_img_path = words[0].replace('profile', 'depth')
            ir_img_path = words[0].replace('profile', 'ir')
            imgs.append((words[0], depth_img_path, ir_img_path, int(words[1])))
                
        self.imgs = imgs
        self.transform = conf.train.transform
        self.transform1 = conf.train.transform1
        self.transform2 = conf.train.transform2
        self.target_transform = target_transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, fn2, fn3, label = self.imgs[index]
        img1 = self.loader_rgb(os.path.join(self.root,fn1))
        img2 = self.loader_gray(os.path.join(self.root,fn2))
        img3 = self.loader_gray(os.path.join(self.root,fn3))

        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))
        # img2 = img2.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))
        # img3 = img3.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform1(img2)
            img3 = self.transform2(img3)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return [img1,img2,img3], label

    def __len__(self):
        return len(self.imgs)

class MyDataset_huoti_test(Dataset):
    def __init__(self, conf, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        imgs = []
        self.root = conf.data_folder
        with open(conf.test_list, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                all_images = os.listdir(os.path.join(self.root, line.split()[0], 'profile'))
                for val in all_images:
                    # imgs.append((os.path.join(line.split()[0], 'profile', val), int(line.split()[1])))
                    rgb_img = os.path.join(line, 'profile', val)
                    depth_img = rgb_img.replace('profile', 'depth')
                    ir_img = rgb_img.replace('profile', 'ir')
                    imgs.append((rgb_img, depth_img, ir_img))

        self.imgs = imgs
        self.transform = conf.test.transform
        self.transform1 = conf.test.transform1
        self.transform2 = conf.test.transform2
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder

    def __getitem__(self, index):
        fn1, fn2, fn3= self.imgs[index]
        img11 = self.loader_rgb(os.path.join(self.root,fn1))
        img12 = self.loader_gray(os.path.join(self.root,fn2))
        img13 = self.loader_gray(os.path.join(self.root,fn3))
        # size_c = (8, 8, 120, 120)
        # img11 = img11.crop(size_c)
        # img12 = img12.crop(size_c)
        # img13 = img13.crop(size_c)

        img21 = self.loader_rgb(os.path.join(self.root, fn1)).transpose(Image.FLIP_LEFT_RIGHT)
        img22 = self.loader_gray(os.path.join(self.root, fn2)).transpose(Image.FLIP_LEFT_RIGHT)
        img23 = self.loader_gray(os.path.join(self.root, fn3)).transpose(Image.FLIP_LEFT_RIGHT)
        # img21 = img21.crop(size_c)
        # img22 = img22.crop(size_c)
        # img23 = img23.crop(size_c)

        if self.transform is not None:
            img11 = self.transform(img11)
            img12 = self.transform1(img12)
            img13 = self.transform2(img13)
            img21 = self.transform(img21)
            img22 = self.transform1(img22)
            img23 = self.transform2(img23)
        return [img11,img12,img13,img21,img22,img23], [fn1, fn2, fn3]

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_rgb(Dataset):
    def __init__(self, conf, target_transform=None, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        imgs = []
        self.conf = conf
        if conf.depth:
            with open(conf.train_list, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in lines:
                    imgs.append((line.split()[0].replace('profile', 'depth'), int(line.split()[1])))
        elif conf.ir:
            with open(conf.train_list, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in lines:
                    imgs.append((line.split()[0].replace('profile', 'ir'), int(line.split()[1])))
        else:
            with open(conf.train_list, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in lines:
                    imgs.append((line.split()[0], int(line.split()[1])))      
        self.imgs = imgs
        self.transform = conf.train.transform
        self.transform1 = conf.train.transform1
        self.target_transform = target_transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        if self.conf.depth:
            img1 = self.loader_gray(os.path.join(self.root, fn1))
        else:
            img1 = self.loader_rgb(os.path.join(self.root, fn1))


        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            if self.conf.depth:
                img1 = self.transform1(img1)
            else:
                img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return [img1], label

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_test_rgb(Dataset):
    def __init__(self, conf, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        imgs = []
        self.conf = conf
        self.root = conf.data_folder
        if conf.depth:
            with open(conf.test_list, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in lines:
                    all_images = os.listdir(os.path.join(self.root, line.split()[0], 'depth'))
                    for val in all_images:
                        # imgs.append((os.path.join(line.split()[0], 'profile', val), int(line.split()[1])))
                        imgs.append(os.path.join(line, 'depth', val))
        elif conf.ir:
            with open(conf.test_list, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in lines:
                    all_images = os.listdir(os.path.join(self.root, line.split()[0], 'ir'))
                    for val in all_images:
                        # imgs.append((os.path.join(line.split()[0], 'profile', val), int(line.split()[1])))
                        imgs.append(os.path.join(line, 'ir', val))
        else:
            with open(conf.test_list, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                for line in lines:
                    all_images = os.listdir(os.path.join(self.root, line.split()[0], 'profile'))
                    for val in all_images:
                        # imgs.append((os.path.join(line.split()[0], 'profile', val), int(line.split()[1])))
                        imgs.append(os.path.join(line, 'profile', val))
        

        self.imgs = imgs
        self.transform = conf.test.transform
        self.transform1 = conf.test.transform1
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        

    def __getitem__(self, index):
        fn1= self.imgs[index]
        if self.conf.depth:
            img11 = self.loader_gray(os.path.join(self.root,fn1))
        else:
            img11 = self.loader_rgb(os.path.join(self.root,fn1))


        if self.transform is not None:
            if self.conf.depth:
                img11 = self.transform1(img11)
            else:   
                img11 = self.transform(img11)

        return [img11], fn1

    def __len__(self):
        return len(self.imgs)


def process_method_5(data_dict, expand_ratio=1.0, f='train', extra_expand_ratio = 1.125):
    img = cv2.imdecode(np.fromfile(data_dict['img_path'], dtype=np.uint8), -1)
    if img is None:
        assert False, 'image `{}` is empty.'.format(data_dict['img_path'])
    rects = copy.deepcopy(data_dict['rects'])
   # print(img.shape)
    if rects == [-1.0] * 4 or rects[0] < 0 or rects[1] < 0 or rects[2] < 0 or rects[3] < 0:
        # print('not need to crop')
        # print(data_dict['img_path'])
        img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_final = Image.fromarray(img_final.astype('uint8'))
        return img_final
    else:
        img_h, img_w = img.shape[:2]

        w = rects[2] - rects[0]
        h = rects[3] - rects[1]
        if h > w:
            origin = rects[0] + rects[2]
            rects[0] = np.maximum((origin / 2. - h / 2.), 0)
            rects[2] = np.minimum((origin / 2. + h / 2.), img_w)
        else:
            origin = rects[1] + rects[3]
            rects[1] = np.maximum((origin / 2. - w / 2.), 0)
            rects[3] = np.minimum((origin / 2. + w / 2.), img_h)

        w = rects[2] - rects[0]
        h = rects[3] - rects[1]
        expand_ratio = expand_ratio
        bbox1 = [np.maximum(rects[0] - w * (expand_ratio - 1) / 2., 0),
                np.maximum(rects[1] - h * (expand_ratio - 1) / 2., 0),
                np.minimum(rects[2] + w * (expand_ratio - 1) / 2., img_w),
                np.minimum(rects[3] + h * (expand_ratio - 1) / 2., img_h)]
        
        if f == 'test':
           # print('test')
            bbox1 = [int(x) for x in bbox1]
            img_final = img[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
           # print(img_final.shape)
            try:
                img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
                img_final = Image.fromarray(img_final.astype('uint8'))
            except:
                print(bbox1)
                print(data_dict['img_path'])
            return img_final
        

        rec_w = bbox1[2] - bbox1[0]
        rec_h = bbox1[3] - bbox1[1]
        rec_expand_ratio = extra_expand_ratio
        bbox2 = [np.maximum(bbox1[0] - rec_w * (rec_expand_ratio - 1) / 2., 0),
                np.maximum(bbox1[1] - rec_h * (rec_expand_ratio - 1) / 2., 0),
                np.minimum(bbox1[2] + rec_w * (rec_expand_ratio - 1) / 2., img_w),
                np.minimum(bbox1[3] + rec_h * (rec_expand_ratio - 1) / 2., img_h)]
        
        bbox2 = [int(x) for x in bbox2]
        img_final = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
        img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
        img_final = Image.fromarray(img_final.astype('uint8'))
    # img_final = cv2.resize(img_final, (144, 144))
        return img_final 


class MyDataset_huoti_rgb_crop(Dataset):
    def __init__(self, conf, target_transform=None, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        self.root = conf.data_folder
        self.conf = conf
        imgs = []
        with open(conf.train_list, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                data_dict = {}
                data = line.split()                
                img_path = data[0]
                label = int(data[-1])
                try:
                    rects = [float(x) for x in data[-5:-1]]
                except:
                    continue
                data_dict['img_path'] = os.path.join(self.root, img_path)
                data_dict['rects'] = rects
                imgs.append((data_dict, label))      
        self.imgs = imgs
        self.transform = conf.train.transform
        self.transform1 = conf.train.transform1
        self.target_transform = target_transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset
        self.extra_expand_ratio = conf.model.input_size / (conf.model.input_size - conf.model.random_offset)

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        # img1 = self.loader_rgb(os.path.join(self.root,fn1))
        img1 = process_method_5(fn1, extra_expand_ratio=self.extra_expand_ratio)


        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
          #  img1 = cv2.flip(img1, 1)

        if self.transform is not None:
            if self.conf.depth:
                img1 = self.transform1(img1)
            else:
                img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return [img1], label

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_rgb_test_crop(Dataset):
    def __init__(self, conf, target_transform=None, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        self.root = conf.data_folder
        self.conf = conf
        imgs = []
        with open(conf.test_list, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                data_dict = {}
                data = line.split()                
                img_path = data[0]
                label = int(data[-1])
                try:
                    rects = [float(x) for x in data[-5:-1]]
                except:
                    continue
                data_dict['img_path'] = os.path.join(self.root, img_path)
                data_dict['rects'] = rects
                imgs.append((data_dict, label))      
        self.imgs = imgs
        self.transform = conf.test.transform
        self.transform1 = conf.test.transform1
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        # img1 = self.loader_rgb(os.path.join(self.root,fn1))
        img1 = process_method_5(fn1, f='test')
        img_path = fn1['img_path']

        if self.transform is not None:
            if self.conf.depth:
                img1 = self.transform1(img1)
            else:
                img1 = self.transform(img1)
        
        return [img1], img_path

    def __len__(self):
        return len(self.imgs)