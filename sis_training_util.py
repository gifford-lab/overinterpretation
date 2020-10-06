"""Util for training on SIS/backselect subsets."""

import glob
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

import sis_analysis_util
import sis_util
from sufficient_input_subsets import sis


def load_preds_file(preds_filepath):
    preds_df = pd.read_csv(
        preds_filepath,
        header=None,
        names=['dataset_idx', 'pred_class_idx', 'confidence'],
    )
    return preds_df


def create_random_mask(shape, pixels_to_keep):
    """Create random mask of pixels.

    If masking all channels per pixel, use `shape` (1, 32, 32).
    """
    mask = np.zeros(shape, dtype=bool)
    # Set pixels_to_keep pixels of mask to True (uniformly at random).
    mask_flat = np.ravel(mask, order='C')
    true_idxs = np.random.choice(
        mask_flat.shape[0], replace=False, size=pixels_to_keep)
    mask_flat[true_idxs] = True
    mask = mask_flat.reshape(shape, order='C')
    assert mask.sum() == pixels_to_keep
    return mask


class BackselectSubsetsCIFARDataset(torch.utils.data.Dataset):
    """Dataset of backselect pixel subsets on CIFAR.

    Args:
        sis_dir (str): Path to directory containing SIS.
        original_dataset: Dataset object containing original images.
        frac_to_keep (float): Fraction of each images to retain (rest masked).
        fully_masked_image (array): Array containing fully masked images.
            Shape should be either (3, 32, 32) or (1, 32, 32) (or broadcastable
            to images).
        true_labels (bool): If True, use true labels as targets for each image
            (rather than predicted labels). If False, must specify
            `preds_filepath`. Default is True.
        preds_filepath (str): Path to file containing predictions on original
            images. See `inference.py` on format details and to generate preds
            file for saved models. Default is None.
        transform (function): Transform to apply to final images.
        target_transform (function): Transform to apply to targets.
    """

    def __init__(self, sis_dir, original_dataset, frac_to_keep,
                 fully_masked_image, true_labels=True, preds_filepath=None,
                 transform=None, target_transform=None):
        self.sis_dir = sis_dir
        self.original_dataset = original_dataset
        self.frac_to_keep = frac_to_keep
        self.fully_masked_image = fully_masked_image
        self.pixels_to_keep = int(np.ceil(1024 * self.frac_to_keep))
        self.true_labels = true_labels
        self.preds_filepath = preds_filepath
        self.transform = transform
        self.target_transform = target_transform
        self.masked_images = None
        self.targets = None

        if not (preds_filepath or true_labels):
            raise ValueError(
                'If not specifying `preds_filepath`, must use `true_labels`.')
        if preds_filepath and true_labels:
            raise ValueError(
                'Must specify only one of `preds_filepath` and `true_labels`.')

        # Load labels from predictions or true labels.
        self._load_labels()

        # Load masks from SIS data.
        self._initialize_masked_images()

        assert self.masked_images.shape[0] == len(self.original_dataset)
        assert self.masked_images.shape[0] == self.targets.shape[0]

    def _load_labels(self):
        if self.true_labels:
            labels = np.array(self.original_dataset.targets)
        else:
            preds_df = load_preds_file(self.preds_filepath)
            labels = preds_df['pred_class_idx'].values
        self.targets = labels

    def _initialize_masked_images(self):
        masked_images = []
        for i in tqdm(range(len(self.original_dataset))):
            image, _ = self.original_dataset[i]
            image = image.cpu().numpy()
            matched_files = glob.glob(
                os.path.join(self.sis_dir, '*_%d.npz' % i))
            assert len(matched_files) == 1
            sr = sis_util.load_sis_result(matched_files[0])
            bs_mask = sis_analysis_util.backselect_mask_from_sis_result(
                sr, self.pixels_to_keep)
            img_masked = sis.produce_masked_inputs(
                image, self.fully_masked_image, [bs_mask])[0]
            masked_images.append(img_masked)
        masked_images = np.array(masked_images)
        self.masked_images = masked_images

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.masked_images[index]
        img = torch.from_numpy(img)
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class RandomSubsetsCIFARDataset(torch.utils.data.Dataset):
    """Dataset of random pixel subsets on CIFAR.

    Random subsets are retained in the dataset so that they remain the same
    each time an image is called. Can be seeded so that subsets are the same
    across initializations of RandomSubsetsCIFARDataset.

    Args:
        original_dataset: Dataset object containing original images.
        frac_to_keep (float): Fraction of each images to retain (rest masked).
        fully_masked_image (array): Array containing fully masked images.
            Shape should be either (3, 32, 32) or (1, 32, 32) (or broadcastable
            to images).
        random_seed (int): Random seed. Default is None (no seed).
        transform (function): Transform to apply to final images.
        target_transform (function): Transform to apply to targets.
    """
    # Random seed constants for train/test splits. Different seeds so the
    #   random subsets are different for the splits.
    # Using the seeding in training scripts enables different models to
    #   train on the same random subsets.
    RANDOM_SEED_TRAIN = 1234
    RANDOM_SEED_TEST = 5678

    def __init__(self, original_dataset, frac_to_keep, fully_masked_image,
                 random_seed=None, transform=None, target_transform=None):
        self.original_dataset = original_dataset
        self.frac_to_keep = frac_to_keep
        self.fully_masked_image = fully_masked_image
        self.random_seed = random_seed
        self.pixels_to_keep = int(np.ceil(1024 * self.frac_to_keep))
        self.transform = transform
        self.target_transform = target_transform
        self.masked_images = None

        if random_seed is not None:
            np.random.seed(random_seed)

        # Use same labels as original dataset.
        self.targets = np.array(self.original_dataset.targets)

        # Create random masked images. Store the masked images so that random
        # subsets remain the same each time an image is called.
        self._initialize_masked_images()

        assert self.masked_images.shape[0] == len(self.original_dataset)
        assert self.masked_images.shape[0] == self.targets.shape[0]

    def _initialize_masked_images(self):
        masked_images = []
        for i in tqdm(range(len(self.original_dataset))):
            image, _ = self.original_dataset[i]
            image = image.cpu().numpy()
            random_mask = create_random_mask((1, 32, 32), self.pixels_to_keep)
            img_masked = np.where(random_mask, image, self.fully_masked_image)
            masked_images.append(img_masked)
        masked_images = np.array(masked_images)
        self.masked_images = masked_images

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.masked_images[index]
        img = torch.from_numpy(img)
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
