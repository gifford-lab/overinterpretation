"""Runs SIS on a sample of CIFAR test image using a pre-trained PyTorch model.

If SIS threshold is 0, runs backward selection which is stored in SISResult.

Example usage:
python run_sis_on_cifar.py \
  --saved_model_dir=./saved_models/resnet18_rep3 \
  --dataset=cifar10_test \
  --start_idx=0 \
  --end_idx=128 \
  --sis_threshold=0

Example on ensemble:
python run_sis_on_cifar.py \
  --saved_model_dir=./saved_models/resnet18_ensemble \
  --dataset=cifar10_test \
  --start_idx=0 \
  --end_idx=128 \
  --sis_threshold=0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

import additional_datasets
import inference_util
import sis_util
import util.misc_util as misc_util
import util.data_util as data_util
from sufficient_input_subsets import sis


DATASET_OPTIONS = [
    'cifar10_train',
    'cifar10_test',
    'cifar100_train',
    'cifar100_test',
    'cifar10c_sample',
]


parser = argparse.ArgumentParser()
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help='Path to saved model directory')
parser.add_argument('--dataset', required=True, choices=DATASET_OPTIONS,
                    help='Dataset name')
parser.add_argument('--start_idx', type=int, required=True,
                    help='Start idx into dataset (inclusive)')
parser.add_argument('--end_idx', type=int, required=True,
                    help='End idx into dataset (inclusive)')
parser.add_argument('--sis_threshold', type=float, default=0.0,
                    help='SIS threshold (default: 0 to run backward selection on all images)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

device = torch.device('cuda' if args.cuda else 'cpu')
print('Using device: ', device)
print()

# Load model from checkpoint.
model = inference_util.load_saved_model(args.saved_model_dir)
model.to(device)
model.eval()
print('Loaded model')

# Load dataset.
if args.dataset == 'cifar10_train':
    transform = data_util.cifar_test_transform()  # No augmentation
    dataset = datasets.CIFAR10(root='data/',
                               train=True,
                               transform=transform,
                               download=True)
elif args.dataset == 'cifar10_test':
    transform = data_util.cifar_test_transform()
    dataset = datasets.CIFAR10(root='data/',
                               train=False,
                               transform=transform,
                               download=True)
elif args.dataset == 'cifar100_train':
    transform = data_util.cifar_test_transform()  # No augmentation
    dataset = datasets.CIFAR100(root='data/',
                                train=True,
                                transform=transform,
                                download=True)
elif args.dataset == 'cifar100_test':
    transform = data_util.cifar_test_transform()
    dataset = datasets.CIFAR100(root='data/',
                                train=False,
                                transform=transform,
                                download=True)
elif args.dataset == 'cifar10c_sample':
    transform = data_util.cifar_test_transform()
    dataset = additional_datasets.CIFAR10C(
        root_dir='data/CIFAR-10-C',
        sis_sample=True,
        transform=transform,
    )
else:
    raise ValueError('Unknown dataset: %s' % args.dataset)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=128,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

# Run SIS on specified test images and write to disk.
sis_out_dir = os.path.join(args.saved_model_dir, 'sis', args.dataset)
print('SIS out directory: ', sis_out_dir)
misc_util.create_directory(sis_out_dir)

initial_mask = sis.make_empty_boolean_mask_broadcast_over_axis([3, 32, 32], 0)
fully_masked_image = np.zeros((3, 32, 32), dtype='float32')

for i in tqdm(range(args.start_idx, args.end_idx+1)):
    sis_outfile = os.path.join(sis_out_dir, '%s_%d.npz' % (args.dataset, i))
    if os.path.exists(sis_outfile):
        continue  # File already exists.
    image, label = dataset[i]
    sis_result = sis_util.find_sis_on_input(
        model, image, initial_mask, fully_masked_image, args.sis_threshold,
        add_softmax=True)
    if sis_result is None:  # No SIS exists.
        continue
    sis_util.save_sis_result(sis_result, sis_outfile)

print('Done')
