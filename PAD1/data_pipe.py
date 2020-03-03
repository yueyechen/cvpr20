import torch
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from my_dataset import MyDataset_huoti_val_patch, MyDataset_huoti_train_rectified, MyDataset_huoti_val_rectified,\
    MyDataset_huoti_val_patch_rectified, MyDataset_huoti_test_rectified


def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def get_train_dataset_huoti_rectified(conf):
    print('train dataset: {}'.format(conf.train_list))
    ds = MyDataset_huoti_train_rectified(conf)
    return ds

def get_val_dataset_huoti_rectified(conf):
    print('val dataset: {}'.format(conf.val_list))
    ds = MyDataset_huoti_val_rectified(conf)
    return ds

def get_test_dataset_huoti_rectified(conf):
    print('val dataset: {}'.format(conf.val_list))
    ds = MyDataset_huoti_test_rectified(conf)
    return ds

def get_train_loader(conf):
    if conf.data_mode == 'huoti':
        ds = get_train_dataset_huoti_rectified(conf)
        if conf.train.sampling:
            from torch.utils.data import WeightedRandomSampler
            weights = [conf.train.sampling_neg if label == 0 else 1 for data, label in ds.imgs]
            train_sampler = WeightedRandomSampler(weights, len(ds), replacement=True)
            loader = DataLoader(ds,
                                batch_size=conf.batch_size,
                                pin_memory=conf.pin_memory,
                                num_workers=conf.num_workers,
                                sampler = train_sampler)
        else:
            flag=True
            loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=flag, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
            print('train dataset shuffle: {}'.format(str(flag)))
    return loader


def get_val_loader(conf):
    if conf.data_mode == 'huoti':
        ds = get_val_dataset_huoti_rectified(conf)
        flag = False
        loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=flag, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)
        print('val dataset shuffle: {}'.format(str(flag)))
    return loader

def get_test_loader(conf):
    if conf.data_mode == 'huoti':
        ds = get_test_dataset_huoti_rectified(conf)
        flag = False
        loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=flag, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)
        print('val dataset shuffle: {}'.format(str(flag)))
    return loader