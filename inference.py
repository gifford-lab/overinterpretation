"""Runs inference using pre-trained model and writes predictions to disk.

Outfile (csv) contains rows of the form:
    dataset_idx,pred_class_idx,confidence

Example Usage:
python inference.py \
    --saved_model_dir=./saved_models/resnet18_rep3 \
    --dataset=cifar10_test
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

import inference_util
import util.data_util as data_util
import util.misc_util as misc_util


DATASET_OPTIONS = [
    'cifar10_train', 'cifar10_test', 'cifar100_train', 'cifar100_test']


parser = argparse.ArgumentParser()
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help='Path to saved model directory')
parser.add_argument('--dataset', required=True, choices=DATASET_OPTIONS,
                    help='Dataset name')
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
else:
    raise ValueError('Unknown dataset: %s' % args.dataset)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=128,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

pred_class_idxs = []
confidences = []
for images, _ in tqdm(iter(data_loader)):
    images = images.to(device)
    batch_preds = inference_util.predict(
        model, images, add_softmax=True).cpu().numpy()
    batch_pred_classes = batch_preds.argmax(axis=1)
    batch_pred_confidences = batch_preds.max(axis=1)
    pred_class_idxs.append(batch_pred_classes)
    confidences.append(batch_pred_confidences)

pred_class_idxs = np.concatenate(pred_class_idxs)
confidences = np.concatenate(confidences)

# Write predictions to disk.
preds_out_dir = os.path.join(args.saved_model_dir, 'preds')
print('Predictions out directory: ', preds_out_dir)
misc_util.create_directory(preds_out_dir)
preds_outfile = os.path.join(preds_out_dir, '%s.csv' % args.dataset)
print('Predictions out file: ', preds_outfile)

with open(preds_outfile, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i, (pred_class, confidence) in enumerate(zip(pred_class_idxs,
                                                     confidences)):
        writer.writerow([i, pred_class, confidence])

print('Done')
