"""Util for PyTorch models."""

from model.resnet import ResNet18
from model.resnet_cifar import resnet20
from model.wide_resnet import WideResNet
from model.vgg import vgg16_bn


def wrn28_10(**kwargs):
    return WideResNet(depth=28, widen_factor=10, dropRate=0.3, **kwargs)


_MODEL_NAME_TO_MODEL = {
    'resnet18': ResNet18,
    'resnet20': resnet20,
    'wideresnet': wrn28_10,
    'vgg16': vgg16_bn,
}


def get_model_from_name(model_name):
    if model_name not in _MODEL_NAME_TO_MODEL:
        raise KeyError('Model not found, got: %s' % model_name)
    return _MODEL_NAME_TO_MODEL[model_name]
