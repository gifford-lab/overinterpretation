"""Util for training on SIS/backselect subsets."""

import glob
import numpy as np
import os
import torch
import torchvision

import imagenet_backselect


def _get_basename_without_extension(filepath):
    return os.path.basename(filepath).split('.')[0]


class BackselectSubsetsImageNetDataset(torchvision.datasets.ImageNet):
    """Dataset of backselect pixel subsets on ImageNet.

    Args:
        sis_dir (str): Path to directory containing SIS.
        imagenet_root: Path to original images dataset root dir.
        frac_to_keep (float): Fraction of each images to retain (rest masked).
        fully_masked_image (array): Array containing fully masked images.
            Shape should be broadcastable to images).
        transform (function): Transform to apply to final images.
        target_transform (function): Transform to apply to targets.
    """

    def __init__(self, sis_dir, imagenet_root, frac_to_keep,
                 fully_masked_image, transform=None, target_transform=None,
                 **kwargs):
        self.sis_dir = sis_dir
        self.imagenet_root = imagenet_root
        self.frac_to_keep = frac_to_keep
        self.fully_masked_image = fully_masked_image
        self.image_to_backselect_file = None

        super(BackselectSubsetsImageNetDataset, self).__init__(
            root=imagenet_root,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )

        self._preprocess_sis_dir()
        assert len(self.image_to_backselect_file) == len(self.samples)

    def _preprocess_sis_dir(self):
        """Initializes map from image filename to backselect file."""
        image_to_backselect_file = {}
        bs_file_format = os.path.join(self.sis_dir, '**', '*.npz')
        for image_path in glob.glob(bs_file_format, recursive=True):
            image_basename = _get_basename_without_extension(image_path)
            image_to_backselect_file[image_basename] = image_path
        self.image_to_backselect_file = image_to_backselect_file

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super(
            BackselectSubsetsImageNetDataset, self).__getitem__(index)

        # Load backselect data and mask image.
        img_path = self.samples[index][0]
        img_basename = _get_basename_without_extension(img_path)
        backselect_filepath = self.image_to_backselect_file[img_basename]
        backselect_result = imagenet_backselect.BackselectResult.from_file(
            backselect_filepath)
        num_iters = backselect_result.mask_order.max()
        mask_after_iter = int((1 - self.frac_to_keep) * num_iters)
        # print(num_iters, mask_after_iter)
        mask = backselect_result.mask_order >= mask_after_iter
        mask = torch.from_numpy(mask)
        # TODO: use self.fully_masked_image
        img = img * mask

        return img, target
