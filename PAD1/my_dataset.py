from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os
import csv
import random
import numpy as np
# from torchvision import transforms as trans


def default_loader(path):
    # return Image.open(path).convert('L')
    return Image.open(path).convert('RGB')

def default_loader_half(path):
    img = Image.open(path).convert('RGB')
    return img.crop((0,0,img.size[0], img.size[1]/2)) #

class MyDataset_huoti(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1], words[2], int(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, fn2, fn3, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))
        img2 = self.loader(os.path.join(str(self.root),fn2))
        img3 = self.loader(os.path.join(str(self.root),fn3))
        # resize and randomCrop
        img1 = img1.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        img2 = img2.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        img3 = img3.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        offset_x = random.randint(0,self.random_offset[0])
        offset_y = random.randint(0,self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        img2 = img2.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        img3 = img3.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15,15)
            img1 = img1.rotate(degree, expand=False)
            img2 = img2.rotate(degree, expand=False)
            img3 = img3.rotate(degree, expand=False)


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # for debug
        # img1_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn1)
        # if not os.path.exists(os.path.dirname(img1_path)):
        #     os.makedirs(os.path.dirname(img1_path))
        # img1.save(img1_path)
        # img2_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn2)
        # if not os.path.exists(os.path.dirname(img2_path)):
        #     os.makedirs(os.path.dirname(img2_path))
        # img2.save(img2_path)
        # img3_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn3)
        # if not os.path.exists(os.path.dirname(img3_path)):
        #     os.makedirs(os.path.dirname(img3_path))
        # img3.save(img3_path)
        # img1 = temp_transform(img1)
        # img2 = temp_transform(img2)
        # img3 = temp_transform(img3)
        return [img1,img2,img3], label, [fn1,fn2,fn3]

    def __len__(self):
        return len(self.imgs)

class MyDataset_huoti_rgb(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], float(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))
        # resize and randomCrop
        img1 = img1.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        offset_x = random.randint(0,self.random_offset[0])
        offset_y = random.randint(0,self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15,15)
            img1 = img1.rotate(degree, expand=False)


        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # for debug
        # img1_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn1)
        # if not os.path.exists(os.path.dirname(img1_path)):
        #     os.makedirs(os.path.dirname(img1_path))
        # img1.save(img1_path)
        # img2_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn2)
        # if not os.path.exists(os.path.dirname(img2_path)):
        #     os.makedirs(os.path.dirname(img2_path))
        # img2.save(img2_path)
        # img3_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn3)
        # if not os.path.exists(os.path.dirname(img3_path)):
        #     os.makedirs(os.path.dirname(img3_path))
        # img3.save(img3_path)
        # img1 = temp_transform(img1)
        # img2 = temp_transform(img2)
        # img3 = temp_transform(img3)
        return [img1], label, [fn1]

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_nir(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[2], int(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))
        # resize and randomCrop
        img1 = img1.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        offset_x = random.randint(0,self.random_offset[0])
        offset_y = random.randint(0,self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15,15)
            img1 = img1.rotate(degree, expand=False)

        # # for debug
        # if random.random() > 0.8:
        #     if label == 1:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/train_pos', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)
        #     elif label == 0:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/train_neg', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)


        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # img2_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn2)
        # if not os.path.exists(os.path.dirname(img2_path)):
        #     os.makedirs(os.path.dirname(img2_path))
        # img2.save(img2_path)
        # img3_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn3)
        # if not os.path.exists(os.path.dirname(img3_path)):
        #     os.makedirs(os.path.dirname(img3_path))
        # img3.save(img3_path)
        # img1 = temp_transform(img1)
        # img2 = temp_transform(img2)
        # img3 = temp_transform(img3)
        return [img1], label, [fn1]

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_depth(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[1], int(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))
        # resize and randomCrop
        img1 = img1.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        offset_x = random.randint(0,self.random_offset[0])
        offset_y = random.randint(0,self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15,15)
            img1 = img1.rotate(degree, expand=False)


        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # for debug
        # img1_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn1)
        # if not os.path.exists(os.path.dirname(img1_path)):
        #     os.makedirs(os.path.dirname(img1_path))
        # img1.save(img1_path)
        # img2_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn2)
        # if not os.path.exists(os.path.dirname(img2_path)):
        #     os.makedirs(os.path.dirname(img2_path))
        # img2.save(img2_path)
        # img3_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn3)
        # if not os.path.exists(os.path.dirname(img3_path)):
        #     os.makedirs(os.path.dirname(img3_path))
        # img3.save(img3_path)
        # img1 = temp_transform(img1)
        # img2 = temp_transform(img2)
        # img3 = temp_transform(img3)
        return [img1], label, [fn1]

    def __len__(self):
        return len(self.imgs)

class MyDataset_huoti_rgb_mix_nir(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[3])))
            imgs.append((words[2], int(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1,label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))
        # resize and randomCrop
        img1 = img1.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        offset_x = random.randint(0,self.random_offset[0])
        offset_y = random.randint(0,self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15,15)
            img1 = img1.rotate(degree, expand=False)


        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # for debug
        # img1_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn1)
        # if not os.path.exists(os.path.dirname(img1_path)):
        #     os.makedirs(os.path.dirname(img1_path))
        # img1.save(img1_path)
        # img2_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn2)
        # if not os.path.exists(os.path.dirname(img2_path)):
        #     os.makedirs(os.path.dirname(img2_path))
        # img2.save(img2_path)
        # img3_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn3)
        # if not os.path.exists(os.path.dirname(img3_path)):
        #     os.makedirs(os.path.dirname(img3_path))
        # img3.save(img3_path)
        # img1 = temp_transform(img1)
        # img2 = temp_transform(img2)
        # img3 = temp_transform(img3)
        return [img1], label, [fn1]

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_rgb_nir_pair(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[2], int(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, fn3, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))
        img3 = self.loader(os.path.join(str(self.root),fn3))
        # resize and randomCrop
        img1 = img1.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        img3 = img3.resize((self.input_size[0]+self.random_offset[0], self.input_size[1]+self.random_offset[1]))
        offset_x = random.randint(0,self.random_offset[0])
        offset_y = random.randint(0,self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        img3 = img3.crop((offset_x, offset_y, offset_x+self.input_size[0], offset_y+self.input_size[1]))
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15,15)
            img1 = img1.rotate(degree, expand=False)
            img3 = img3.rotate(degree, expand=False)


        if self.transform is not None:
            img1 = self.transform(img1)
            img3 = self.transform(img3)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # for debug
        # img1_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn1)
        # if not os.path.exists(os.path.dirname(img1_path)):
        #     os.makedirs(os.path.dirname(img1_path))
        # img1.save(img1_path)
        # img2_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn2)
        # if not os.path.exists(os.path.dirname(img2_path)):
        #     os.makedirs(os.path.dirname(img2_path))
        # img2.save(img2_path)
        # img3_path = os.path.join('/home2/xuejiachen/PAD_Pytorch/work_space/temp', fn3)
        # if not os.path.exists(os.path.dirname(img3_path)):
        #     os.makedirs(os.path.dirname(img3_path))
        # img3.save(img3_path)
        # img1 = temp_transform(img1)
        # img2 = temp_transform(img2)
        # img3 = temp_transform(img3)
        return [img1,img3], label, [fn1,fn3]

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_val(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        # with open(conf.val_list) as f:
        #     f_csv = csv.reader(f)
        #     _ = next(f_csv)
        #     imgs = []
        #     for row in f_csv:
        #         imgs.append((row[0], row[1], row[2], int(row[3])))
        fh = open(conf.val_list, 'r')
        imgs = []
        if conf.eval.format == 'rgb':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0],float(words[3])))
        elif conf.eval.format == 'nir':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[2],int(words[3])))
        elif conf.eval.format == 'depth':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[1],int(words[3])))

        self.imgs = imgs
        self.transform = conf.eval.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.eval.input_size
        self.random_offset = conf.eval.random_offset

    def __getitem__(self, index):
        fn1, label= self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root),fn1))

        img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        left = self.random_offset[0]/2
        top = self.random_offset[1]/2
        right = left + self.input_size[0]
        bottom = top + self.input_size[1]
        img1 = img1.crop((left, top,right, bottom))


        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # # for debug
        # if random.random() > 0.8:
        #     if label == 1:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_pos', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)
        #     elif label == 0:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_neg', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)


        return [img1], label, [fn1]

    def __len__(self):
        return len(self.imgs)

def TTA_5_cropps(img, target_size):
    width, height = img.size
    target_w, target_h = target_size

    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    starts = [
        [start_x, start_y],
        [start_x - target_w, start_y],
        [start_x, start_y - target_w],
        [start_x + target_w, start_y],
        [start_x, start_y + target_w]
    ]

    crops = []
    for start_index in starts:
        x, y = start_index
        x = min(max(0, x), width - target_w - 1)
        y = min(max(0, y), height - target_h - 1)

        patch = img.crop((x, y, x+target_w, y + target_h))
        crops.append(patch)
    return crops

def TTA_9_cropps(img, target_size):
    width, height = img.size
    target_w, target_h = target_size

    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_h],
              [start_x + target_w, start_y],
              [start_x, start_y + target_h],

              [start_x + target_w, start_y + target_h],
              [start_x - target_w, start_y - target_h],
              [start_x - target_w, start_y + target_h],
              [start_x + target_w, start_y - target_h],
              ]

    crops = []
    for start_index in starts:
        x, y = start_index
        x = min(max(0, x), width - target_w - 1)
        y = min(max(0, y), height - target_h - 1)

        patch = img.crop((x, y, x + target_w, y + target_h))
        crops.append(patch)
    return crops

def TTA_18_cropps(img, target_size):
    width, height = img.size
    target_w, target_h = target_size

    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_h],
              [start_x + target_w, start_y],
              [start_x, start_y + target_h],

              [start_x + target_w, start_y + target_h],
              [start_x - target_w, start_y - target_h],
              [start_x - target_w, start_y + target_h],
              [start_x + target_w, start_y - target_h],
              ]

    crops = []
    for start_index in starts:
        x, y = start_index
        x = min(max(0, x), width - target_w - 1)
        y = min(max(0, y), height - target_h - 1)

        patch = img.crop((x, y, x + target_w, y + target_h))
        crops.append(patch)
        crops.append(patch.transpose(Image.FLIP_LEFT_RIGHT))

    return crops

def TTA_36_cropps(img, target_size):
    width, height = img.size
    target_w, target_h = target_size

    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_h],
              [start_x + target_w, start_y],
              [start_x, start_y + target_h],

              [start_x + target_w, start_y + target_h],
              [start_x - target_w, start_y - target_h],
              [start_x - target_w, start_y + target_h],
              [start_x + target_w, start_y - target_h],
              ]

    crops = []
    for start_index in starts:
        x, y = start_index
        x = min(max(0, x), width - target_w - 1)
        y = min(max(0, y), height - target_h - 1)

        patch = img.crop((x, y, x + target_w, y + target_h))
        patch_lr = patch.transpose(Image.FLIP_LEFT_RIGHT)
        crops.append(patch)
        crops.append(patch.transpose(Image.FLIP_TOP_BOTTOM))
        crops.append(patch_lr)
        crops.append(patch_lr.transpose(Image.FLIP_TOP_BOTTOM))

    return crops

class MyDataset_huoti_val_patch(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        # with open(conf.val_list) as f:
        #     f_csv = csv.reader(f)
        #     _ = next(f_csv)
        #     imgs = []
        #     for row in f_csv:
        #         imgs.append((row[0], row[1], row[2], int(row[3])))
        fh = open(conf.val_list, 'r')
        imgs = []
        if conf.eval.format == 'rgb':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[3])))
        elif conf.eval.format == 'nir':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[2], int(words[3])))
        elif conf.eval.format == 'depth':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[1], int(words[3])))

        self.imgs = imgs
        self.transform = conf.eval.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.eval.input_size
        self.random_offset = conf.eval.random_offset
        self.patch_size = conf.patch_size
        self.patch_num = conf.patch_num

    def __getitem__(self, index):
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root), fn1))

        img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))

        if self.patch_num == 5:
            imgs = TTA_5_cropps(img1, self.patch_size)
        elif self.patch_num == 9:
            imgs = TTA_9_cropps(img1, self.patch_size)
        elif self.patch_num == 18:
            imgs = TTA_18_cropps(img1, self.patch_size)
        elif self.patch_num == 36:
            imgs = TTA_36_cropps(img1, self.patch_size)

        # left = self.random_offset[0] / 2
        # top = self.random_offset[1] / 2
        # right = left + self.input_size[0]
        # bottom = top + self.input_size[1]
        # img1 = img1.crop((left, top, right, bottom))

        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # # for debug
        # if random.random() > 0.8:
        #     if label == 1:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_pos', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)
        #     elif label == 0:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_neg', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)

        if self.transform is not None:
            imgs = [self.transform(t) for t in imgs]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return [imgs], label, [fn1]

    def __len__(self):
        return len(self.imgs)


class MyDataset_huoti_train_rectified(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.train_list, 'r')
        imgs = []
        self.rects = []
        self.counter = 0
        if conf.train.format == 'rgb':
            for line in fh:
                data = line.strip().split()
                rgb_name = data[0]
                label = float(data[-1])
                rect = [int(float(x)) for x in data[1:5]]
                if (np.array(rect)==-1).any():
                    continue
                self.counter += 1
                imgs.append((rgb_name, label))
                self.rects.append(rect)
        elif conf.train.format == 'depth':
            for line in fh:
                data = line.strip().split()
                depth_name = data[5]
                label = float(data[-1])
                rect = [int(float(x)) for x in data[6:10]]
                if (np.array(rect)==-1).any():
                    continue
                self.counter += 1
                imgs.append((depth_name, label))
                self.rects.append(rect)
        elif conf.train.format == 'nir':
            for line in fh:
                data = line.strip().split()
                # nir_name = data[10]
                nir_name = data[6]
                label = float(data[-1])
                # rect = [int(float(x)) for x in data[11:15]]
                rect = [int(float(x)) for x in data[7:11]]
                if (np.array(rect)==-1).any():
                    continue
                self.counter += 1
                imgs.append((nir_name, label))
                self.rects.append(rect)
        else:
            raise ValueError

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset
        self.expand_ratio = 1.2
        # self.process_method = process_method(conf.process_method)

    def __getitem__(self, index):
        # ========= rect00 =================
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root), fn1))
        rect = self.rects[index]
        rect_w = rect[2]-rect[0]
        rect_h = rect[3]-rect[1]
        w, h = img1.size
        if rect_w < rect_h:
            origin = rect[0]+rect[2]
            rect[0] = int(origin/2 - rect_h/2)
            rect[2] = int(origin/2 + rect_h/2)
            border_l = abs(rect[0]) if rect[0]<0 else 0
            border_r = (rect[2]-w) if rect[2]>w else 0
            img1 = ImageOps.expand(img1, (border_l, 0 , border_r, 0), 0)
            rect[0] = max(0, rect[0])
            rect[2] = rect[0]+rect_h
        else:
            origin = rect[1]+rect[3]
            rect[1] = int(origin/2 - rect_w/2)
            rect[3] = int(origin/2 + rect_w/2)
            border_t = abs(rect[1]) if rect[1]<0 else 0
            border_b = (rect[3]-h) if rect[3]>h else 0
            img1 = ImageOps.expand(img1, (0, border_t, 0, border_b), 0)
            rect[1]=max(0, rect[1])
            rect[3]=rect[0]+rect_w
        img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        offset_x = random.randint(0, self.random_offset[0])
        offset_y = random.randint(0, self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))
        # img1.save('/home/users/jiachen.xue/temp/0223/{}_{}.jpg'.format(label, index))
       # +++++++++++++++++++++++++
       # ============= rect01 ==================
       #  fn1, label = self.imgs[index]
       #  img1 = self.loader(os.path.join(str(self.root), fn1))
       #  rect = self.rects[index]
       #  rect_w = rect[2] - rect[0]
       #  rect_h = rect[3] - rect[1]
       #  w, h = img1.size
       #  if rect_w < rect_h:
       #      origin = rect[0] + rect[2]
       #      rect[0] = max(int(origin / 2 - rect_h / 2), 0)
       #      rect[2] = min(int(origin / 2 + rect_h / 2), w)
       #  else:
       #      origin = rect[1] + rect[3]
       #      rect[1] = max(int(origin / 2 - rect_w / 2), 0)
       #      rect[3] = min(int(origin / 2 + rect_w / 2), h)
       #  img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
       #  img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
       #  offset_x = random.randint(0, self.random_offset[0])
       #  offset_y = random.randint(0, self.random_offset[1])
       #  img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))
        # img1.save('/home/users/jiachen.xue/temp/0223/rect01/train/{}_{}.jpg'.format(label, index))

       # # ++++++++++++++++++++++++++++++
       # ============== rect02 ============
       #  fn1, label = self.imgs[index]
       #  img1 = self.loader(os.path.join(str(self.root), fn1))
       #  # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_src.jpg'.format(label, index))
       #  rect = self.rects[index]
       #  rect_w = rect[2] - rect[0]
       #  rect_h = rect[3] - rect[1]
       #  w, h = img1.size
       #  if rect_w < rect_h:
       #      origin = rect[0] + rect[2]
       #      rect[0] = int(origin / 2 - rect_h / 2)
       #      rect[2] = int(origin / 2 + rect_h / 2)
       #      border_l = abs(rect[0]) if rect[0] < 0 else 0
       #      border_r = (rect[2] - w) if rect[2] > w else 0
       #      img1 = ImageOps.expand(img1, (border_l, 0, border_r, 0), 0)
       #      rect[0] = max(0, rect[0])
       #      rect[2] = rect[0] + rect_h
       #  else:
       #      origin = rect[1] + rect[3]
       #      rect[1] = int(origin / 2 - rect_w / 2)
       #      rect[3] = int(origin / 2 + rect_w / 2)
       #      border_t = abs(rect[1]) if rect[1] < 0 else 0
       #      border_b = (rect[3] - h) if rect[3] > h else 0
       #      img1 = ImageOps.expand(img1, (0, border_t, 0, border_b), 0)
       #      rect[1] = max(0, rect[1])
       #      rect[3] = rect[0] + rect_w
       #  #img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_buqi.jpg'.format(label, index))
       #  w,h = img1.size
       #  rect_len = rect[2]-rect[0]
       #  target_len = int(rect_len*self.expand_ratio)
       #  origin_w = rect[0]+rect[2]
       #  origin_h = rect[1]+rect[3]
       #  rect[0] = int(origin_w/2-target_len/2)
       #  rect[2] = int(origin_w/2+target_len/2)
       #  rect[1] = int(origin_h/2-target_len/2)
       #  rect[3] = int(origin_h/2+target_len/2)
       #  border_l = abs(rect[0]) if rect[0]< 0 else 0
       #  border_r = (rect[2]-w) if rect[2]>w else 0
       #  border_t = abs(rect[1]) if rect[1]<0 else 0
       #  border_b = (rect[3]-h) if rect[3]>h else 0
       #  img1 = ImageOps.expand(img1, (border_l,  border_t, border_r, border_b), 0)
       #  # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_expand.jpg'.format(label, index))
       #  rect[0] = max(0, rect[0])
       #  rect[1] = max(0, rect[1])
       #  rect[2] = rect[0]+target_len
       #  rect[3] = rect[1]+target_len
       #  # +++++++++++++++++++++++++++++
       #
       #  img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
       #  img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
       #  # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_256.jpg'.format(label, index))
       #  offset_x = random.randint(0, self.random_offset[0])
       #  offset_y = random.randint(0, self.random_offset[1])
       #  img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

    # ++++++++++++++++++++++++++++++++++
       #
        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        # random rotate
        if random.random() > 0.2:
            degree = random.randint(-15, 15)
            img1 = img1.rotate(degree, expand=False)

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # left = self.random_offset[0] / 2
        # top = self.random_offset[1] / 2
        # right = left + self.input_size[0]
        # bottom = top + self.input_size[1]
        # img1 = img1.crop((left, top, right, bottom))

        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # # for debug
        # if random.random() > 0.8:
        #     if label == 1:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_pos', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)
        #     elif label == 0:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_neg', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)

        return [img1], label, [fn1]

    def __len__(self):
        return self.counter


class MyDataset_huoti_val_rectified(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.val_list, 'r')
        imgs = []
        self.rects = []
        self.counter = 0
        if conf.eval.format == 'rgb':
            for line in fh:
                data = line.strip().split()
                rgb_name = data[0]
                label = float(data[-1])
                rect = [int(float(x)) for x in data[1:5]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append((rgb_name, label))
                self.rects.append(rect)
        elif conf.eval.format == 'depth':
            for line in fh:
                data = line.strip().split()
                depth_name = data[5]
                label = float(data[-1])
                rect = [int(float(x)) for x in data[6:10]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append((depth_name, label))
                self.rects.append(rect)
        elif conf.eval.format == 'nir':
            for line in fh:
                data = line.strip().split()
                # nir_name = data[10]
                nir_name = data[6]
                label = float(data[-1])
                # rect = [int(float(x)) for x in data[11:15]]
                rect = [int(float(x)) for x in data[7:11]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append((nir_name, label))
                self.rects.append(rect)
        else:
            raise ValueError

        self.imgs = imgs
        self.transform = conf.eval.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.eval.input_size
        self.random_offset = conf.eval.random_offset
        self.expand_ratio = 1.2

    def __getitem__(self, index):
        # =========== rect00 ====================
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root), fn1))
        rect = self.rects[index]
        rect_w = rect[2] - rect[0]
        rect_h = rect[3] - rect[1]
        w, h = img1.size
        if rect_w < rect_h:
            origin = rect[0] + rect[2]
            rect[0] = int(origin / 2 - rect_h / 2)
            rect[2] = int(origin / 2 + rect_h / 2)
            border_l = abs(rect[0]) if rect[0] < 0 else 0
            border_r = (rect[2] - w) if rect[2] > w else 0
            img1 = ImageOps.expand(img1, (border_l, 0, border_r, 0), 0)
            rect[0] = max(0, rect[0])
            rect[2] = rect[0] + rect_h
        else:
            origin = rect[1] + rect[3]
            rect[1] = int(origin / 2 - rect_w / 2)
            rect[3] = int(origin / 2 + rect_w / 2)
            border_t = abs(rect[1]) if rect[1] < 0 else 0
            border_b = (rect[3] - h) if rect[3] > h else 0
            img1 = ImageOps.expand(img1, (0, border_t, 0, border_b), 0)
            rect[1] = max(0, rect[1])
            rect[3] = rect[1] + rect_w
        img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        left = self.random_offset[0] / 2
        top = self.random_offset[1] / 2
        right = left + self.input_size[0]
        bottom = top + self.input_size[1]
        img1 = img1.crop((left, top, right, bottom))
        # ++++++++++++++++++++++++++++
        # ================== rect01 =================
        # fn1, label = self.imgs[index]
        # img1 = self.loader(os.path.join(str(self.root), fn1))
        # rect = self.rects[index]
        # rect_w = rect[2] - rect[0]
        # rect_h = rect[3] - rect[1]
        # w, h = img1.size
        # if rect_w < rect_h:
        #     origin = rect[0] + rect[2]
        #     rect[0] = max(0, int(origin / 2 - rect_h / 2))
        #     rect[2] = min(w, int(origin / 2 + rect_h / 2))
        # else:
        #     origin = rect[1] + rect[3]
        #     rect[1] = max(0, int(origin / 2 - rect_w / 2))
        #     rect[3] = min(h, int(origin / 2 + rect_w / 2))
        # img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        # img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        # left = self.random_offset[0] / 2
        # top = self.random_offset[1] / 2
        # right = left + self.input_size[0]
        # bottom = top + self.input_size[1]
        # img1 = img1.crop((left, top, right, bottom))
        # img1.save('/home/users/jiachen.xue/temp/0223/rect01/val/{}_{}.jpg'.format(label, index))
        # +++++++++++++++++++++++++++++++++++++
        # ============== rect02 ============
        # fn1, label = self.imgs[index]
        # img1 = self.loader(os.path.join(str(self.root), fn1))
        # # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_src.jpg'.format(label, index))
        # rect = self.rects[index]
        # rect_w = rect[2] - rect[0]
        # rect_h = rect[3] - rect[1]
        # w, h = img1.size
        # if rect_w < rect_h:
        #     origin = rect[0] + rect[2]
        #     rect[0] = int(origin / 2 - rect_h / 2)
        #     rect[2] = int(origin / 2 + rect_h / 2)
        #     border_l = abs(rect[0]) if rect[0] < 0 else 0
        #     border_r = (rect[2] - w) if rect[2] > w else 0
        #     img1 = ImageOps.expand(img1, (border_l, 0, border_r, 0), 0)
        #     rect[0] = max(0, rect[0])
        #     rect[2] = rect[0] + rect_h
        # else:
        #     origin = rect[1] + rect[3]
        #     rect[1] = int(origin / 2 - rect_w / 2)
        #     rect[3] = int(origin / 2 + rect_w / 2)
        #     border_t = abs(rect[1]) if rect[1] < 0 else 0
        #     border_b = (rect[3] - h) if rect[3] > h else 0
        #     img1 = ImageOps.expand(img1, (0, border_t, 0, border_b), 0)
        #     rect[1] = max(0, rect[1])
        #     rect[3] = rect[0] + rect_w
        # # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_buqi.jpg'.format(label, index))
        # w, h = img1.size
        # rect_len = rect[2] - rect[0]
        # target_len = int(rect_len * self.expand_ratio)
        # origin_w = rect[0] + rect[2]
        # origin_h = rect[1] + rect[3]
        # rect[0] = int(origin_w / 2 - target_len / 2)
        # rect[2] = int(origin_w / 2 + target_len / 2)
        # rect[1] = int(origin_h / 2 - target_len / 2)
        # rect[3] = int(origin_h / 2 + target_len / 2)
        # border_l = abs(rect[0]) if rect[0] < 0 else 0
        # border_r = (rect[2] - w) if rect[2] > w else 0
        # border_t = abs(rect[1]) if rect[1] < 0 else 0
        # border_b = (rect[3] - h) if rect[3] > h else 0
        # img1 = ImageOps.expand(img1, (border_l, border_t, border_r, border_b), 0)
        # # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_expand.jpg'.format(label, index))
        # rect[0] = max(0, rect[0])
        # rect[1] = max(0, rect[1])
        # rect[2] = rect[0] + target_len
        # rect[3] = rect[1] + target_len
        #
        # img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        # img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        # left = self.random_offset[0] / 2
        # top = self.random_offset[1] / 2
        # right = left + self.input_size[0]
        # bottom = top + self.input_size[1]
        # img1 = img1.crop((left, top, right, bottom))
        # +++++++++++++++++++++++++++++

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # left = self.random_offset[0] / 2
        # top = self.random_offset[1] / 2
        # right = left + self.input_size[0]
        # bottom = top + self.input_size[1]
        # img1 = img1.crop((left, top, right, bottom))

        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # # for debug
        # if random.random() > 0.8:
        #     if label == 1:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_pos', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)
        #     elif label == 0:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_neg', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)

        return [img1], label, [fn1]

    def __len__(self):
        return self.counter

class MyDataset_huoti_val_patch_rectified(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        # with open(conf.val_list) as f:
        #     f_csv = csv.reader(f)
        #     _ = next(f_csv)
        #     imgs = []
        #     for row in f_csv:
        #         imgs.append((row[0], row[1], row[2], int(row[3])))
        fh = open(conf.val_list, 'r')
        imgs = []
        self.rects = []
        self.counter = 0
        if conf.eval.format == 'rgb':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                rect = [int(float(x)) for x in words[1:-1]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append((words[0], int(words[-1])))
                self.rects.append([int(float(x)) for x in words[1:-1]])
        elif conf.eval.format == 'nir':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[2], int(words[3])))
        elif conf.eval.format == 'depth':
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[1], int(words[3])))

        self.imgs = imgs
        self.transform = conf.eval.transform
        self.target_transform = target_transform
        if conf.model.half_face:
            self.loader = default_loader_half
        else:
            self.loader = loader
        self.root = conf.huoti_folder
        self.input_size = conf.eval.input_size
        self.random_offset = conf.eval.random_offset
        self.patch_size = conf.patch_size
        self.patch_num = conf.patch_num
        self.expand_ratio = 1.2


    def __getitem__(self, index):
        # =========== rect00 ====================
        # fn1, label = self.imgs[index]
        # img1 = self.loader(os.path.join(str(self.root), fn1))
        # rect = self.rects[index]
        # rect_w = rect[2] - rect[0]
        # rect_h = rect[3] - rect[1]
        # w, h = img1.size
        # if rect_w < rect_h:
        #     origin = rect[0] + rect[2]
        #     rect[0] = int(origin / 2 - rect_h / 2)
        #     rect[2] = int(origin / 2 + rect_h / 2)
        #     border_l = abs(rect[0]) if rect[0] < 0 else 0
        #     border_r = (rect[2] - w) if rect[2] > w else 0
        #     img1 = ImageOps.expand(img1, (border_l, 0, border_r, 0), 0)
        #     rect[0] = max(0, rect[0])
        #     rect[2] = rect[0] + rect_h
        # else:
        #     origin = rect[1] + rect[3]
        #     rect[1] = int(origin / 2 - rect_w / 2)
        #     rect[3] = int(origin / 2 + rect_w / 2)
        #     border_t = abs(rect[1]) if rect[1] < 0 else 0
        #     border_b = (rect[3] - h) if rect[3] > h else 0
        #     img1 = ImageOps.expand(img1, (0, border_t, 0, border_b), 0)
        #     rect[1] = max(0, rect[1])
        #     rect[3] = rect[1] + rect_w
        # img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        # img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        # ++++++++++++++++++++++++++++
        # ================== rect01 =================
        fn1, label = self.imgs[index]
        img1 = self.loader(os.path.join(str(self.root), fn1))
        rect = self.rects[index]
        rect_w = rect[2] - rect[0]
        rect_h = rect[3] - rect[1]
        w, h = img1.size
        if rect_w < rect_h:
            origin = rect[0] + rect[2]
            rect[0] = max(0, int(origin / 2 - rect_h / 2))
            rect[2] = min(w, int(origin / 2 + rect_h / 2))
        else:
            origin = rect[1] + rect[3]
            rect[1] = max(0, int(origin / 2 - rect_w / 2))
            rect[3] = min(h, int(origin / 2 + rect_w / 2))
        img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        # img1.save('/home/users/jiachen.xue/temp/0223/rect01/val/{}_{}.jpg'.format(label, index))
        # +++++++++++++++++++++++++++++++++++++
        # ============== rect02 ============
        # fn1, label = self.imgs[index]
        # img1 = self.loader(os.path.join(str(self.root), fn1))
        # # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_src.jpg'.format(label, index))
        # rect = self.rects[index]
        # rect_w = rect[2] - rect[0]
        # rect_h = rect[3] - rect[1]
        # w, h = img1.size
        # if rect_w < rect_h:
        #     origin = rect[0] + rect[2]
        #     rect[0] = int(origin / 2 - rect_h / 2)
        #     rect[2] = int(origin / 2 + rect_h / 2)
        #     border_l = abs(rect[0]) if rect[0] < 0 else 0
        #     border_r = (rect[2] - w) if rect[2] > w else 0
        #     img1 = ImageOps.expand(img1, (border_l, 0, border_r, 0), 0)
        #     rect[0] = max(0, rect[0])
        #     rect[2] = rect[0] + rect_h
        # else:
        #     origin = rect[1] + rect[3]
        #     rect[1] = int(origin / 2 - rect_w / 2)
        #     rect[3] = int(origin / 2 + rect_w / 2)
        #     border_t = abs(rect[1]) if rect[1] < 0 else 0
        #     border_b = (rect[3] - h) if rect[3] > h else 0
        #     img1 = ImageOps.expand(img1, (0, border_t, 0, border_b), 0)
        #     rect[1] = max(0, rect[1])
        #     rect[3] = rect[0] + rect_w
        # # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_buqi.jpg'.format(label, index))
        # # expand ratio
        # w, h = img1.size
        # rect_len = rect[2] - rect[0]
        # target_len = int(rect_len * self.expand_ratio)
        # origin_w = rect[0] + rect[2]
        # origin_h = rect[1] + rect[3]
        # rect[0] = int(origin_w / 2 - target_len / 2)
        # rect[2] = int(origin_w / 2 + target_len / 2)
        # rect[1] = int(origin_h / 2 - target_len / 2)
        # rect[3] = int(origin_h / 2 + target_len / 2)
        # border_l = abs(rect[0]) if rect[0] < 0 else 0
        # border_r = (rect[2] - w) if rect[2] > w else 0
        # border_t = abs(rect[1]) if rect[1] < 0 else 0
        # border_b = (rect[3] - h) if rect[3] > h else 0
        # img1 = ImageOps.expand(img1, (border_l, border_t, border_r, border_b), 0)
        # # img1.save('/home/users/jiachen.xue/temp/0224/rect02_1.1/{}_{}_expand.jpg'.format(label, index))
        # rect[0] = max(0, rect[0])
        # rect[1] = max(0, rect[1])
        # rect[2] = rect[0] + target_len
        # rect[3] = rect[1] + target_len
        #
        # img1 = img1.crop((rect[0], rect[1], rect[2], rect[3]))
        # img1 = img1.resize((self.input_size[0] + self.random_offset[0], self.input_size[1] + self.random_offset[1]))
        # +++++++++++++++++++++++++++++

        if self.patch_num == 5:
            imgs = TTA_5_cropps(img1, self.patch_size)
        elif self.patch_num == 9:
            imgs = TTA_9_cropps(img1, self.patch_size)
        elif self.patch_num == 18:
            imgs = TTA_18_cropps(img1, self.patch_size)
        elif self.patch_num == 36:
            imgs = TTA_36_cropps(img1, self.patch_size)

        # left = self.random_offset[0] / 2
        # top = self.random_offset[1] / 2
        # right = left + self.input_size[0]
        # bottom = top + self.input_size[1]
        # img1 = img1.crop((left, top, right, bottom))

        # offset_x = random.randint(0, self.random_offset[0])
        # offset_y = random.randint(0, self.random_offset[1])
        # img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # # for debug
        # if random.random() > 0.8:
        #     if label == 1:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_pos', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)
        #     elif label == 0:
        #         img1_path = os.path.join('/home/users/jiachen.xue/PAD_Pytorch/work_space/temp1024/val_neg', fn1)
        #         if not os.path.exists(os.path.dirname(img1_path)):
        #             os.makedirs(os.path.dirname(img1_path))
        #         img1.save(img1_path)

        if self.transform is not None:
            imgs = [self.transform(t) for t in imgs]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return [imgs], label, [fn1]

    def __len__(self):
        return self.counter