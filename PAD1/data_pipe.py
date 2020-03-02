import torch
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from my_dataset import MyDataset_huoti_rgb, MyDataset_huoti_rgb_mix_nir, MyDataset_huoti_depth, \
    MyDataset_huoti_rgb_nir_pair, MyDataset_huoti_val, MyDataset_huoti_nir, \
    MyDataset_huoti_val_patch, MyDataset_huoti_train_rectified, MyDataset_huoti_val_rectified,\
    MyDataset_huoti_val_patch_rectified


def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def get_train_dataset_huoti(conf):
    print('train dataset: {}'.format(conf.train_list))
    if conf.train.format == 'rgb':
        ds = MyDataset_huoti_rgb(conf)
    elif conf.train.format == 'nir':
        ds = MyDataset_huoti_nir(conf)
    elif conf.train.format == 'depth':
        ds = MyDataset_huoti_depth(conf)
    elif conf.train.format == 'rgb_mix_nir':
        ds = MyDataset_huoti_rgb_mix_nir(conf)
    elif conf.train.format == 'rgb_nir_pair':
        ds = MyDataset_huoti_rgb_nir_pair(conf)
    else:
        assert False, 'only support conf.train.format = \'rgb\', \'rgb_mix_nir\', \'rgb_nir_pair\''
    return ds

def get_train_dataset_huoti_rectified(conf):
    print('train dataset: {}'.format(conf.train_list))
    ds = MyDataset_huoti_train_rectified(conf)
    return ds

def get_val_dataset_huoti_rectified(conf):
    print('val dataset: {}'.format(conf.val_list))
    ds = MyDataset_huoti_val_rectified(conf)
    return ds

def get_val_dataset_huoti(conf):
    print('val dataset: {}'.format(conf.val_list))
    ds = MyDataset_huoti_val(conf)
    return ds

def get_train_loader(conf):
    if conf.data_mode == 'huoti':
        if conf.train.get('rectified', 'False'):
            ds = get_train_dataset_huoti_rectified(conf)
        else:
            ds = get_train_dataset_huoti(conf)
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
        if conf.train.get('rectified', False):
            ds = get_val_dataset_huoti_rectified(conf)
        else:
            ds = get_val_dataset_huoti(conf)
        flag = False
        loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=flag, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)
        print('val dataset shuffle: {}'.format(str(flag)))
    return loader
