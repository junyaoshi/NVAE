# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""
import random
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
import pickle
from tqdm import tqdm
from lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN


class SomethingSomething(Dataset):
    def __init__(self, data_dir, train=True, transform=None, debug=False):
        """
        Args:
            data_dir = '/scratch/agelosk/Hands/something_something_paths.pkl'
        """
        self.transform = transform
        self.train = train
        # get all demo directories

        self.image_paths = []
        self.hand_paths = []
        with open(data_dir, 'rb') as f:
            temp = pickle.load(f)
        random.shuffle(temp)
        self.image_paths, self.hand_paths = zip(*temp)

        if self.train:
            print("Total size of dataset:", len(self.image_paths), "images")

        if self.train:
            self.image_paths = self.image_paths[0:int(len(self.image_paths) * 0.9)]
            self.hand_paths = self.hand_paths[0:int(len(self.hand_paths) * 0.9)]
        else:
            self.image_paths = self.image_paths[int(len(self.image_paths) * 0.9):]
            self.hand_paths = self.hand_paths[int(len(self.hand_paths) * 0.9):]
        if debug:
            if self.train:
                self.image_paths = self.image_paths[0:100]
                self.hand_paths = self.hand_paths[0:100]
            else:
                self.image_paths = self.image_paths[100:200]
                self.hand_paths = self.hand_paths[100:200]

        assert len(self.image_paths) == len(self.hand_paths)

        if self.train:
            print("Training set has:     ", len(self.image_paths), "images")
        else:
            print("Validation set has:   ", len(self.image_paths), "images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        hand_path = self.hand_paths[idx]
        img = Image.open(image_path)
        w, h = img.size

        bbs = pickle.load(open(hand_path, 'rb'))
        hand = bbs['pred_output_list'][0]['left_hand'] if len(bbs['pred_output_list'][0]['left_hand']) > 0 \
            else bbs['pred_output_list'][0]['right_hand']
        hand_bb = bbs['hand_bbox_list'][0]['left_hand'] if bbs['hand_bbox_list'][0]['left_hand'] is not None \
            else bbs['hand_bbox_list'][0]['right_hand']
        params_3d = hand['pred_joints_smpl'].reshape(63)
        # cropped_hand = np.array(img.crop((hand_bb[0],hand_bb[1],hand_bb[0]+hand_bb[2],hand_bb[1]+hand_bb[3])))

        cropped_hand = np.zeros((np.array(img).shape), np.uint8)
        cropped_hand[hand_bb[1]:hand_bb[1] + hand_bb[3], hand_bb[0]:hand_bb[0] + hand_bb[2]] = \
            np.array(img)[hand_bb[1]:hand_bb[1] + hand_bb[3], hand_bb[0]:hand_bb[0] + hand_bb[2]]
        cropped_hand = Image.fromarray(cropped_hand)

        if self.transform is not None:
            image = self.transform(img)
            cropped_hand = self.transform(cropped_hand)
        hand_bb = np.array([hand_bb[0] / w, hand_bb[1] / h, hand_bb[2] / w, hand_bb[3] / h]).astype(np.float32)

        # Dataloader returns:
        #   1. image: Frame from something-something dataset of shape given by self.transform
        #   2. params_3d: 63 features produced by 3D Hand Reconstruction
        #   3. hand_bb: 4 numbers in the form [x,y,w,h] that indicate the position of the hand in the image.
        #   4. cropped_hand: Image of the cropped hand according to hand_bb padded on the size of the image.
        return image, params_3d, hand_bb, cropped_hand


class XMagicalDataset(Dataset):
    def __init__(self, data_dir, transform=None, debug=False):
        """
        Args:
            data_dir: for example, /home/junyao/Datasets/xirl/xmagical/train
        """
        self.transform = transform
        # get all demo directories
        robot_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        demo_dirs = [f.path for robot_dir in robot_dirs for f in os.scandir(robot_dir) if f.is_dir()]
        if debug:
            demo_dirs = demo_dirs[:3]

        self.image_paths = []
        self.robot_states = []
        self.world_states = []
        self.robot_types = []
        t = 3  # number of debris (targets)
        # loop over them, in each loop gather list of image paths and list of robot states
        for demo_dir in tqdm(demo_dirs, desc=f'Processing data from {data_dir}...'):
            states_json = os.path.join(demo_dir, 'states.json')
            with open(states_json, 'r') as f:
                states = json.load(f)  # states are 3-frame stack
            states = states[1:]  # first, ignore the first frame to make len(states) and len(images) match
            states = [s[len(s) // 3 * 2:] for s in states]  # take only the last frame of the 3-frame stack
            robot_states = [[s[0], s[1], s[2 * t + 2], s[2 * t + 3]] for s in states]  # x, y, cos(t), sin(t) of robot
            world_states = [s[2:(2 * t + 2)] for s in states]  # x, y of each debris (3 in total)
            image_paths = sorted(
                # glob.glob(os.path.join(demo_dir, '*.png')),
                [f for f in os.listdir(demo_dir) if f.endswith('.png')],
                key=lambda fname: int(fname.split('.')[0])
            )
            n_samples = len(image_paths)
            image_paths = [os.path.join(demo_dir, path) for path in image_paths]
            if 'gripper' in demo_dir:
                robot_type = [1., 0., 0., 0.]
            elif 'longstick' in demo_dir:
                robot_type = [0., 1., 0., 0.]
            elif 'mediumstick' in demo_dir:
                robot_type = [0., 0., 1., 0.]
            else:
                robot_type = [0., 0., 0., 1.]
            robot_types = [robot_type for _ in range(n_samples)]

            self.robot_states.extend(robot_states)
            self.world_states.extend(world_states)
            self.image_paths.extend(image_paths)
            self.robot_types.extend(robot_types)
        self.robot_states = np.array(self.robot_states)
        self.world_states = np.array(self.world_states)
        self.robot_types = np.array(self.robot_types)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        robot_state = self.robot_states[idx].astype(np.float32)
        world_state = self.world_states[idx].astype(np.float32)
        robot_type = self.robot_types[idx].astype(np.float32)

        rs_x, rs_y, rs_c, rs_s = robot_state  # robot x, y, cos, sin
        ws_x1, ws_y1, ws_x2, ws_y2, ws_x3, ws_y3 = world_state # world x1, y1, ..., x3, y3
        n = np.random.choice([0, 1, 2, 3])
        # rotate 0 degree
        if n == 0:
            image_rot = image
            robot_state_rot = robot_state
            world_state_rot = world_state

        # rotate 90 degrees
        elif n == 1:
            image_rot = torch.rot90(image, k=1, dims=(2, 1))
            robot_state_rot = np.array([rs_y, -rs_x, rs_s, -rs_c])
            world_state_rot = np.array([ws_y1, -ws_x1, ws_y2, -ws_x2, ws_y3, -ws_x3])

        # rotate 180 degrees
        elif n == 2:
            image_rot = torch.rot90(image, k=2, dims=(2, 1))
            robot_state_rot = np.array([-rs_x, -rs_y, -rs_c, -rs_s])
            world_state_rot = np.array([-ws_x1, -ws_y1, -ws_x2, -ws_y2, -ws_x3, -ws_y3])

        # rotate 270 degrees
        else:
            image_rot = torch.rot90(image, k=1, dims=(1, 2))
            robot_state_rot = np.array([-rs_y, rs_x, -rs_s, rs_c])
            world_state_rot = np.array([-ws_y1, ws_x1, -ws_y2, ws_x2, -ws_y3, ws_x3])

        return image_rot, robot_state_rot, world_state_rot, robot_type

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
        train_transform, valid_transform = _data_transforms_xmagical(resize)
        train_data = XMagicalDataset(
            data_dir=os.path.join(args.data, 'train'), transform=train_transform, debug=args.debug
        )
        valid_data = XMagicalDataset(
            data_dir=os.path.join(args.data, 'valid'), transform=valid_transform, debug=args.debug
        )
    elif dataset == 'something-something':
        num_classes = 0
        resize = 64
        train_transform, valid_transform = _data_transforms_something_something(resize)
        train_data = SomethingSomething(data_dir=args.data, train=True, transform=train_transform, debug=args.debug)
        valid_data = SomethingSomething(data_dir=args.data, train=False, transform=valid_transform, debug=args.debug)
    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    print('Creating data loaders...')
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=0 if args.debug else args.num_workers, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=0 if args.debug else args.num_workers, drop_last=True)
    print('Creating data loaders: done')

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


def _data_transforms_xmagical(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_something_something(size):
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


if __name__ == '__main__':
    data_dir = '/home/junyao/Datasets/xirl/xmagical'
    num_classes = 0
    resize = 64
    train_transform, valid_transform = _data_transforms_xmagical(resize)
    train_data = XMagicalDataset(data_dir=os.path.join(data_dir, 'train'), transform=train_transform, debug=True)
    print(f'train length: {len(train_data)}')
    valid_data = XMagicalDataset(data_dir=os.path.join(data_dir, 'valid'), transform=valid_transform, debug=True)
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
        sampler=valid_sampler, pin_memory=True, num_workers=0, drop_last=True)

    import matplotlib.pyplot as plt
    mask_threshold = 0.91
    for idx, data in enumerate(train_queue):
        image, robot_state, world_state, robot_type = data

        # test robot mask (mask out everything but the robot)
        fig, axes = plt.subplots(3, 1, figsize=(5, 15))
        sample_image = image[0]
        mask = sample_image[0] < mask_threshold

        # ignore grey pixels on the edge
        mask[0] = False
        mask[-1] = False
        mask[:, 0] = False
        mask[:, -1] = False

        mask = mask.repeat(3, 1, 1)
        masked_image = sample_image * mask
        axes[0].imshow(sample_image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('original image')
        axes[1].imshow(mask.float().permute(1, 2, 0).cpu().numpy())
        axes[1].set_title('mask')
        axes[2].imshow(masked_image.permute(1, 2, 0).cpu().numpy())
        axes[2].set_title('masked image')
        x, y, c, s = robot_state[0]
        x1, y1, x2, y2, x3, y3 = world_state[0]
        fig.suptitle(f'x: {x:.3f} | y: {y:.3f} | c: {c:.3f} | s: {s:.3f}\n'
                     f'ws: {x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}, {x3:.3f}, {y3:.3f}')
        plt.tight_layout()
        plt.show()
        plt.close()
        if idx == 10:
            break
