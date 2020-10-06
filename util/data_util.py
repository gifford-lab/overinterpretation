"""Util for datasets and transforms."""

import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt


CIFAR_MEAN = np.array([125.3, 123.0, 113.9])
CIFAR_STD = np.array([63.0, 62.1, 66.7])

CIFAR_NORMALIZE = transforms.Normalize(mean=[x / 255.0 for x in CIFAR_MEAN],
                                       std=[x / 255.0 for x in CIFAR_STD])
CIFAR_RANDOM_CROP = transforms.RandomCrop(32, padding=4)
CIFAR_RANDOM_FLIP = transforms.RandomHorizontalFlip()

_DATASET_TO_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
}


def cifar_train_transform(data_augmentation):
    train_transform = transforms.Compose([])
    if data_augmentation:
        train_transform.transforms.append(CIFAR_RANDOM_CROP)
        train_transform.transforms.append(CIFAR_RANDOM_FLIP)
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(CIFAR_NORMALIZE)
    return train_transform


def cifar_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        CIFAR_NORMALIZE,
    ])
    return test_transform


def unnorm_cifar_image(x):
    # Unnormalize image by undoing mean/std normalization.
    # Image is kept in the range [0, 1].
    x_unnorm = x * (CIFAR_STD / 255.0) + (CIFAR_MEAN / 255.0)
    x_unnorm = np.clip(x_unnorm, 0, 1)
    return x_unnorm


def transpose_channel_last(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if len(x.shape) == 4:  # Includes batch dimension.
        order = (0, 2, 3, 1)
    else:
        order = (1, 2, 0)
    return np.transpose(x, order)


def plot_cifar_image(x, unnorm=False):
    if x.shape[0] == 3:  # Transpose channel last.
        x = transpose_channel_last(x)
    if unnorm:
        x = unnorm_cifar_image(x)
    plt.imshow(x)
    plt.axis('off')


def get_num_classes_for_dataset(dataset):
    if dataset not in _DATASET_TO_NUM_CLASSES:
        raise ValueError('Unknown dataset: %s' % dataset)
    return _DATASET_TO_NUM_CLASSES[dataset]
