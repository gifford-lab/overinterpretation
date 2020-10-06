"""Utils for loading and evaluating ImageNet models."""

import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from util import misc_util


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def load_trained_imagenet_model(saved_model_dir):
    config = misc_util.load_config(misc_util.get_config_path(saved_model_dir))
    arch = config['arch']
    if arch.startswith('inception'):
        model = models.__dict__[arch](aux_logits=False)
    else:
        model = models.__dict__[arch]()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint_path = os.path.join(
        saved_model_dir, 'checkpoints', 'checkpoint.pth.tar')
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    print('Loaded model from checkpoint.')
    return model


def get_transforms(arch):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize_pixels = 256
    center_crop_pixels = 224
    if 'inception' in arch:
        resize_pixels = 299
        center_crop_pixels = 299
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(center_crop_pixels),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_pixels),
        transforms.CenterCrop(center_crop_pixels),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform



def accuracy(loader, net, cuda=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total


def accuracy_topk(loader, net, topk=(1,), cuda=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    avg_meters = [AverageMeter('top-%d' % k) for k in topk]

    with torch.no_grad():
        for data in loader:
            images, target = data
            if cuda:
                images = images.cuda()
                target = target.cuda()
            output = net(images)

            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i, k in enumerate(topk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                batch_k_acc = correct_k.mul_(100.0 / batch_size)
                avg_meters[i].update(batch_k_acc, images.size(0))

    topk_avgs = np.array([float(am.avg) for am in avg_meters])

    return topk_avgs
