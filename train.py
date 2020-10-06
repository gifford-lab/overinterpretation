"""Trains CNN on CIFAR and writes checkpoint to disk.

Example Usage:
python train.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --data_augmentation \
    --tensorboard \
    --out_dir=saved_models/resnet18_rep1

Training on SIS pixel subsets:
python train.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --data_augmentation \
    --tensorboard \
    --out_dir=saved_models/resnet18_rep3/sis/saved_models/resnet18_rep1 \
    --train_on_sis \
    --frac_to_keep=0.05 \
    --sis_dir=saved_models/resnet18_rep3/sis

Training on random pixel subsets:
python train.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --data_augmentation \
    --tensorboard \
    --out_dir=saved_models/random_subsets/0.05/resnet18_rep1 \
    --train_on_random_subsets \
    --frac_to_keep=0.05

Train with random replacement (input dropout) augmentation:
python train.py \
    --dataset=cifar10 \
    --model=resnet18 \
    --data_augmentation \
    --tensorboard \
    --input_dropout \
    --keep_prob=0.8 \
    --num_samples=1 \
    --out_dir=saved_models/input_dropout/resnet18_input-dropout-0.8_rep1
"""

import argparse
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

import augmentations
import sis_training_util
from util import data_util
from util import misc_util

from model.resnet import ResNet18
from model.wide_resnet import WideResNet
from model.resnet_cifar import resnet20
from model.vgg import vgg16_bn


model_options = ['resnet18', 'wideresnet', 'resnet20', 'vgg16']
dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay (L2 penalty)')

parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')

parser.add_argument('--train_on_sis', action='store_true', default=False,
                    help='train on backselect sis pixel subsets')
parser.add_argument('--train_on_random_subsets', action='store_true', default=False,
                    help='train on random pixel subsets')
parser.add_argument('--frac_to_keep', type=float, default=0.05,
                    help='fraction of pixels in each image to retain')
parser.add_argument('--sis_dir', type=str,
                    help='path to directory containg sis, should contain subdirectories for _train and _test sets')

parser.add_argument('--input_dropout', action='store_true', default=False,
                    help='randomly replaces pixels with zero at input layer')
parser.add_argument('--keep_prob', type=float, default=0.8,
                    help='probability each pixel should be retained for input dropout')
parser.add_argument('--num_samples', type=int, default=1,
                    help='number of random samples to average over for input dropout')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Writes files for tensorboard.')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Out directory (for model checkpoint)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

if args.train_on_sis and args.train_on_random_subsets:
    raise ValueError(
        'Cannot set both --train_on_sis and --train_on_random_subsets.')

if args.seed is not None:
    print('Setting seed: ', args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

print(args)

# Initialize transforms.
train_transform = data_util.cifar_train_transform(args.data_augmentation)
test_transform = data_util.cifar_test_transform()

# Load dataset.
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)

# Wrap dataset if training on backselect or random subsets.
if args.train_on_sis:
    print('Initializing backselect subsets datasets.')
    fully_masked_image = np.zeros((3, 32, 32), dtype='float32')
    train_dataset = sis_training_util.BackselectSubsetsCIFARDataset(
        sis_dir=os.path.join(args.sis_dir, '%s_train' % args.dataset),
        original_dataset=train_dataset,
        true_labels=True,
        preds_filepath=None,
        frac_to_keep=args.frac_to_keep,
        fully_masked_image=fully_masked_image,
    )
    test_dataset = sis_training_util.BackselectSubsetsCIFARDataset(
        sis_dir=os.path.join(args.sis_dir, '%s_test' % args.dataset),
        original_dataset=test_dataset,
        true_labels=True,
        preds_filepath=None,
        frac_to_keep=args.frac_to_keep,
        fully_masked_image=fully_masked_image,
    )
elif args.train_on_random_subsets:
    print('Initializing random subsets datasets.')
    fully_masked_image = np.zeros((3, 32, 32), dtype='float32')
    train_dataset = sis_training_util.RandomSubsetsCIFARDataset(
        original_dataset=train_dataset,
        frac_to_keep=args.frac_to_keep,
        fully_masked_image=fully_masked_image,
        random_seed=(
            sis_training_util.RandomSubsetsCIFARDataset.RANDOM_SEED_TRAIN),
    )
    test_dataset = sis_training_util.RandomSubsetsCIFARDataset(
        original_dataset=test_dataset,
        frac_to_keep=args.frac_to_keep,
        fully_masked_image=fully_masked_image,
        random_seed=(
            sis_training_util.RandomSubsetsCIFARDataset.RANDOM_SEED_TEST),
    )


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                     dropRate=0.3)
elif args.model == 'resnet20':
    cnn = resnet20(num_classes=num_classes)
elif args.model == 'vgg16':
    cnn = vgg16_bn(num_classes=num_classes)


# Wrap model if using input dropout.
if args.input_dropout:
    print('Wrapping model with input dropout.')
    cnn = augmentations.ModelWithInputDropout(
        cnn,
        args.keep_prob,
        num_samples=args.num_samples,
    )


criterion = torch.nn.CrossEntropyLoss()
if args.cuda:
    cnn = cnn.cuda()
    criterion = criterion.cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True,
                                weight_decay=args.weight_decay)
scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

# Initialize log.
log_path = misc_util.get_log_path(args.out_dir)
misc_util.create_directory(os.path.dirname(log_path))
csv_logger = misc_util.CSVLogger(
    fieldnames=['epoch', 'train_acc', 'test_acc'],
    filepath=log_path,
)

# Write args (model configuration) to JSON file.
config_path = misc_util.get_config_path(args.out_dir)
misc_util.write_args_to_json(args, config_path)


# Initialize SummaryWriter for TensorBoard
if args.tensorboard:
    tensorboard_log_dir = os.path.join(args.out_dir, 'runs')
    writer = SummaryWriter(log_dir=tensorboard_log_dir, flush_secs=20)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step()

    row = {
        'epoch': str(epoch),
        'train_acc': str(accuracy),
        'test_acc': str(test_acc),
    }

    csv_logger.writerow(row)

    if args.tensorboard:
        global_step = epoch + 1
        writer.add_scalar('xentropy loss', xentropy_loss_avg / (i + 1),
                          global_step=global_step)
        writer.add_scalar('train acc', accuracy, global_step=global_step)
        writer.add_scalar('test acc', test_acc, global_step=global_step)


# Save model checkpoint.
checkpoint_path = misc_util.get_checkpoint_path(args.out_dir)
misc_util.create_directory(os.path.dirname(checkpoint_path))
torch.save(cnn.state_dict(), checkpoint_path)

# Close logger and SummaryWriter.
csv_logger.close()
if args.tensorboard:
    writer.close()
