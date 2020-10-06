"""Util for data augmentation transforms to correct overinterpretation."""

import numpy as np
import torch


def create_dropout_mask(shape, keep_prob, device=None):
    return torch.rand(*shape, device=device) <= keep_prob


class RandomPixelDropout(object):
    """Randomly replaces a subset of pixels with zeros.

    Args:
        keep_prob (float): Probability each pixel should be retained.
    """
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W). Image should be
                normalized to unit normal.
        Returns:
            Tensor: Image where each pixel is replaced with zeros with
                probability 1 - `keep_prob`. With probability `keep_prob`,
                each pixel retains its original value. Note that the channels
                in each pixel are either all retained or all replaced.
        """
        mask = create_dropout_mask(
            (1, img.shape[1], img.shape[2]), self.keep_prob, device=img.device)
        return mask * img


class ModelWithInputDropout(torch.nn.Module):
    def __init__(self, model, keep_prob, num_samples=1):
        super(ModelWithInputDropout, self).__init__()
        self.model = model
        self.keep_prob = keep_prob
        self.num_samples = num_samples

    # For speedup, assumes that batch size (x.shape[0]) * self.num_samples can
    #   fit into a single GPU batch so all dropout images are run through the
    #   network simultaneously.
    def forward(self, x):
        batch = x.repeat(self.num_samples, 1, 1, 1)
        mask = create_dropout_mask(
            (batch.shape[0], 1, batch.shape[2], batch.shape[3]),
            self.keep_prob,
            device=x.device,
        )
        batch = batch * mask
        outs = self.model(batch)
        outs = outs.reshape(self.num_samples, x.shape[0], *outs.shape[1:])
        return torch.mean(outs, dim=0)
