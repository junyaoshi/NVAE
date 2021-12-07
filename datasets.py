# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import numpy as np
from PIL import Image
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import json
import urllib
import glob
from tqdm import tqdm
from lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN


class XMagicalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: for example, /home/junyao/Datasets/xirl/xmagical/train
        """
        self.transform = transform
        # get all demo directories
        robot_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        demo_dirs = [f.path for robot_dir in robot_dirs for f in os.scandir(robot_dir) if f.is_dir()]

        self.image_paths = []
        self.robot_states = []
        self.world_states = []
        self.robot_types = []
        # loop over them, in each loop gather list of image paths and list of robot states
        for demo_dir in tqdm(demo_dirs, desc=f'Processing data from {data_dir}...'):
            states_json = os.path.join(demo_dir, 'states.json')
            with open(states_json, 'r') as f:
                states = json.load(f)
            t = len(states[0]) // 4 - 1
            robot_states = [[s[0], s[1], s[2*t+2], s[2*t+3]] for s in states]
            world_states = [s[2:(2+2*t)] for s in states]
            image_paths = sorted(
                # glob.glob(os.path.join(demo_dir, '*.png')),
                [f for f in os.listdir(demo_dir) if f.endswith('.png')],
                key=lambda fname: int(fname.split('.')[0])
            )
            image_paths = [os.path.join(demo_dir, path) for path in image_paths]
            if 'gripper' in demo_dir:
                robot_type = [1., 0., 0., 0.]
            elif 'longstick' in demo_dir:
                robot_type = [0., 1., 0., 0.]
            elif 'mediumstick' in demo_dir:
                robot_type = [0., 0., 1., 0.]
            else:
                robot_type = [0., 0., 0., 1.]

            self.robot_states.extend(robot_states)
            self.world_states.extend(world_states)
            self.image_paths.extend(image_paths)
            self.robot_types.extend(robot_type)
        self.robot_states = np.array(self.robot_states)
        self.world_states = np.array(self.world_states)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        robot_state = self.robot_states[idx]
        world_state = self.world_states[idx]
        robot_type = self.robot_types[idx]
        return image, robot_state, world_state, robot_type

class StackedMNIST(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(StackedMNIST, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        index1 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index2 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index3 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        self.num_images = 2 * len(self.data)

        self.index = []
        for i in range(self.num_images):
            self.index.append((index1[i], index2[i], index3[i]))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        target = 0
        for i in range(3):
            img_, target_ = self.data[self.index[index][i]], int(self.targets[self.index[index][i]])
            img[:, :, i] = img_
            target += target_ * 10 ** (2 - i)

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args)

def download_omniglot(data_dir):
    filename = 'chardata.mat'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        print('Downloaded', filename)

    return


def load_omniglot(data_dir):
    download_omniglot(data_dir)

    data_path = os.path.join(data_dir, 'chardata.mat')

    omni = loadmat(data_path)
    train_data = 255 * omni['data'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))
    test_data = 255 * omni['testdata'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))

    train_data = train_data.astype('uint8')
    test_data = test_data.astype('uint8')

    return train_data, test_data


class OMNIGLOT(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        d = self.data[index]
        img = Image.fromarray(d)
        return self.transform(img), 0     # return zero as label.

    def __len__(self):
        return len(self.data)

def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'mnist':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_mnist(args)
        train_data = dset.MNIST(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.MNIST(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'stacked_mnist':
        num_classes = 1000
        train_transform, valid_transform = _data_transforms_stacked_mnist(args)
        train_data = StackedMNIST(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = StackedMNIST(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'omniglot':
        num_classes = 0
        download_omniglot(args.data)
        train_transform, valid_transform = _data_transforms_mnist(args)
        train_data, valid_data = load_omniglot(args.data)
        train_data = OMNIGLOT(train_data, train_transform)
        valid_data = OMNIGLOT(valid_data, valid_transform)
    elif dataset.startswith('celeba'):
        if dataset == 'celeba_64':
            resize = 64
            num_classes = 40
            train_transform, valid_transform = _data_transforms_celeba64(resize)
            train_data = LMDBDataset(root=args.data, name='celeba64', train=True, transform=train_transform, is_encoded=True)
            valid_data = LMDBDataset(root=args.data, name='celeba64', train=False, transform=valid_transform, is_encoded=True)
        elif dataset in {'celeba_256'}:
            num_classes = 1
            resize = int(dataset.split('_')[1])
            train_transform, valid_transform = _data_transforms_generic(resize)
            train_data = LMDBDataset(root=args.data, name='celeba', train=True, transform=train_transform)
            valid_data = LMDBDataset(root=args.data, name='celeba', train=False, transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('lsun'):
        if dataset.startswith('lsun_bedroom'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['bedroom_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['bedroom_val'], transform=valid_transform)
        elif dataset.startswith('lsun_church'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['church_outdoor_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['church_outdoor_val'], transform=valid_transform)
        elif dataset.startswith('lsun_tower'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['tower_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['tower_val'], transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet'):
        num_classes = 1
        resize = int(dataset.split('_')[1])
        assert args.data.replace('/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(root=args.data, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='imagenet-oord', train=False, transform=valid_transform)
    elif dataset.startswith('ffhq'):
        num_classes = 1
        resize = 256
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(root=args.data, name='ffhq', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='ffhq', train=False, transform=valid_transform)
    elif dataset == 'xmagical':
        num_classes = 0
        resize = 64
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = XMagicalDataset(data_dir=os.path.join(args.data, 'train'), transform=train_transform)
        valid_data = XMagicalDataset(data_dir=os.path.join(args.data, 'valid'), transform=valid_transform)
    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, num_classes


def _data_transforms_cifar10(args):
    """Get data transforms for cifar10."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, valid_transform


def _data_transforms_mnist(args):
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        Binarize(),
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        Binarize(),
    ])

    return train_transform, valid_transform


def _data_transforms_stacked_mnist(args):
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])

    return train_transform, valid_transform


def _data_transforms_generic(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_lsun(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


if __name__ == '__main__':
    data_dir = '/home/junyao/Datasets/xirl/xmagical'
    num_classes = 0
    resize = 64
    train_transform, valid_transform = _data_transforms_generic(resize)
    train_data = XMagicalDataset(data_dir=os.path.join(data_dir, 'train'), transform=train_transform)
    print(f'train length: {len(train_data)}')
    valid_data = XMagicalDataset(data_dir=os.path.join(data_dir, 'valid'), transform=valid_transform)
    print(f'valid length: {len(valid_data)}')

    train_sampler, valid_sampler = None, None
    batch_size = 8

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=0, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=0, drop_last=False)

    for idx, data in enumerate(train_queue):
        image, robot_state, world_state, robot_type = data
        break
