"""Definition of additional datasets.

Includes CIFAR-10-C test set.
"""

import numpy as np
import os
import torch
from PIL import Image


class CIFAR10C(torch.utils.data.Dataset):
    """CIFAR-10-C Dataset.

    From the paper: https://arxiv.org/abs/1807.01697

    Args:
        root_dir (str): Path to dataset. Extracted CIFAR-10-C.tar from:
            https://zenodo.org/record/2535967#.XkH_AlJKjUI
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    SIS_SAMPLE_RANDOM_SEED = 1234
    SIS_SAMPLE_SIZE = 2000

    def __init__(self, root_dir, which_corruption=None, which_severity=None,
                 sis_sample=False, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.which_corruption = which_corruption
        self.which_severity = which_severity
        self.sis_sample = sis_sample
        self.transform = transform
        self.target_transform = target_transform

        if self.sis_sample and (self.which_corruption or self.which_severity):
            raise ValueError(
                'Cannot set both `sis_sample` or corruption/severity filter.')

        # Load data.
        self.data, self.targets = self._load_all_data()

        # Filter for SIS random sample.
        if self.sis_sample:
            sample_idxs = self._get_sis_sample_idxs()
            self.data = self.data[sample_idxs]
            self.targets = self.targets[sample_idxs]

        assert self.data.shape[0] == self.targets.shape[0]

    @staticmethod
    def get_corruptions():
        return  [
            'brightness',
            'contrast',
            'defocus_blur',
            'elastic_transform',
            'fog',
            'frost',
            'gaussian_blur',
            'gaussian_noise',
            'glass_blur',
            'impulse_noise',
            'jpeg_compression',
            'motion_blur',
            'pixelate',
            'saturate',
            'shot_noise',
            'snow',
            'spatter',
            'speckle_noise',
            'zoom_blur',
        ]

    def _get_sis_sample_idxs(self):
        np.random.seed(self.SIS_SAMPLE_RANDOM_SEED)
        idxs = np.random.choice(
            self.data.shape[0], size=self.SIS_SAMPLE_SIZE, replace=False)
        return idxs

    def _load_data_for_corruption(self, corruption):
        data = np.load(os.path.join(self.root_dir, '%s.npy' % corruption))
        targets = np.load(os.path.join(self.root_dir, 'labels.npy'))
        if self.which_severity and self.which_severity >= 1:
            assert self.which_severity <= 5
            start_idx = (self.which_severity - 1) * 10000
            end_idx = self.which_severity * 10000
            data = data[start_idx:end_idx]
            targets = targets[start_idx:end_idx]

        return data, targets

    def _load_all_data(self):
        data = []
        targets = []
        if self.which_corruption is not None:
            assert self.which_corruption in self.get_corruptions()
            corruptions = [self.which_corruption]
        else:
            corruptions = self.get_corruptions()
        for cor in corruptions:
            cor_data, cor_targets = self._load_data_for_corruption(cor)
            data.append(cor_data)
            targets.append(cor_targets)
        return np.concatenate(data), np.concatenate(targets)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target = self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
