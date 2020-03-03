from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os
import csv
import random
import numpy as np
# from torchvision import transforms as trans


def default_loader(path):
    return Image.open(path).convert('RGB')

def default_loader_half(path):
    img = Image.open(path).convert('RGB')
    return img.crop((0,0,img.size[0], img.size[1]/2)) #

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

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return [img1], label, [fn1]

    def __len__(self):
        return self.counter

class MyDataset_huoti_test_rectified(Dataset):
    def __init__(self, conf, target_transform=None, loader=default_loader):
        fh = open(conf.val_list, 'r')
        imgs = []
        self.rects = []
        self.counter = 0
        if conf.eval.format == 'rgb':
            for line in fh:
                data = line.strip().split()
                rgb_name = data[0]
                rect = [int(float(x)) for x in data[1:5]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append(rgb_name)
                self.rects.append(rect)
        elif conf.eval.format == 'depth':
            for line in fh:
                data = line.strip().split()
                depth_name = data[5]
                rect = [int(float(x)) for x in data[6:10]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append(depth_name)
                self.rects.append(rect)
        elif conf.eval.format == 'nir':
            for line in fh:
                data = line.strip().split()
                # nir_name = data[10]
                nir_name = data[6]
                # rect = [int(float(x)) for x in data[11:15]]
                rect = [int(float(x)) for x in data[7:11]]
                if (np.array(rect) == -1).any():
                    continue
                self.counter += 1
                imgs.append(nir_name)
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
        fn1 = self.imgs[index]
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

        if self.transform is not None:
            img1 = self.transform(img1)

        return [img1], [fn1]

    def __len__(self):
        return self.counter

