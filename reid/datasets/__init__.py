from __future__ import absolute_import
import warnings

from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhk03 import CUHK03
from .cuhk01 import CUHK01
from .cuhk_sysu import CUHK_SYSU
from .viper import VIPeR
from .prid import PRID
from .cuhk02 import CUHK02

import os.path as osp
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data import IterLoader
from reid.utils.data.sampler import RandomIdentitySampler, RandomMultipleGallerySampler

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'cuhk_sysu': CUHK_SYSU,
    'cuhk03': CUHK03,
    'cuhk01': CUHK01,
    'viper': VIPeR,
    'prid': PRID,
    'cuhk02': CUHK02
}

def names():
    return sorted(__factory.keys())

def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)

def get_data(name, data_dir, height, width, batch_size, workers, num_instances):
    if name == "cuhk_sysu":
        root = osp.join(data_dir, "cuhksysu4reid")
    elif name == "msmt17":
        root = osp.join(data_dir, "MSMT17")
    else:
        root = osp.join(data_dir, name)

    dataset = create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(dataset.query + dataset.gallery),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=batch_size, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_loader, test_loader, init_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader