"""Gradient-based backward selection for ImageNet."""

import collections
import numpy as np
import os
import torch


class BackselectResult(
    collections.namedtuple(
        'BackselectResult',
        [
            'original_confidences', 'target_class_idx',
            'confidences_over_backselect', 'mask_order',
        ],
    )):
    """Specifies result of the `run_gradient_backward_selection` procedure.
    """
    __slots__ = ()

    def __len__(self):
        """Defines len as number of backward selection iterations."""
        return confidences_over_backselect.shape[0]

    def __hash__(self):
        return NotImplemented

    def __eq__(self, other):
        """Checks equality between this and another BackselectResult.

        Check that all fields are the exactly equal.

        Args:
            other: A BackselectResult instance.
        Returns:
            True if self and other are equal, and False otherwise.
        """
        if not isinstance(other, BackselectResult):
            return False

        return (np.array_equal(self.original_confidences,
                               other.original_confidences) and
                np.array_equal(self.target_class_idx,
                               other.target_class_idx) and
                np.array_equal(self.confidences_over_backselect,
                               other.confidences_over_backselect) and
                np.array_equal(self.mask_order, other.mask_order))

    def __ne__(self, other):
        return not self == other

    def approx_equal(self, other, rtol=1e-05, atol=1e-08):
        """Checks that this and another BackselectResult are approximately equal.

        BackselectResult.{target_class_idx, mask_order} are compared exactly,
        while BackselectResult.{original_confidences,
        confidences_over_backselect} are compared with slight tolerance (using
        np.allclose with provided rtol and atol). This is intended to check
        equality allowing for small differences due to floating point
        representations.

        Args:
            other: A BackselectResult instance.
            rtol: Float, the relative tolerance parameter used when comparing
                `values_over_entire_backselect` (see documentation for np.allclose).
            atol: Float, the absolute tolerance parameter used when comparing
                `values_over_entire_backselect` (see documentation for np.allclose).
        Returns:
            True if self and other are approximately equal, and False otherwise.
        """
        if not isinstance(other, BackselectResult):
            return False

        # BackselectResult.{target_class_idx, mask_order} compared exactly.
        # BackselectResult.{original_confidences, confidences_over_backselect}
        #    compared with slight tolerance.
        return (np.array_equal(self.target_class_idx,
                               other.target_class_idx) and
                np.array_equal(self.mask_order,
                               other.mask_order) and
                np.allclose(
                    self.original_confidences,
                    other.original_confidences,
                    rtol=rtol,
                    atol=atol) and
               np.allclose(
                    self.confidences_over_backselect,
                    other.confidences_over_backselect,
                    rtol=rtol,
                    atol=atol))

    def to_file(self, filepath, target_confidences_only=False,
                low_precision=False):
        """Writes this BackselectResult to filepath.

        Data is stored in compressed .npz format (see `np.savez_compressed`).

        Args:
            filepath (str): Path to file (should use .npz extension).
            target_confidences_only (boolean): If True, only stores confidences
                toward `self.target_class_idx` in `confidences_over_backselect`
                and `original_confidences`. If False (default), stores fields
                as-is.
            low_precision (boolean): If True, stores `original_confidences` and
                `confidences_over_backselect` as np.float16 and `mask_order` as
                np.uint16. If False (default), stores fields as-is (default
                32-bit).

        Returns:
            None.
        """
        original_confidences = np.copy(self.original_confidences)
        confidences_over_backselect = np.copy(self.confidences_over_backselect)
        mask_order = np.copy(self.mask_order)
        if target_confidences_only:
            original_confidences = original_confidences[self.target_class_idx]
            confidences_over_backselect = confidences_over_backselect[:, self.target_class_idx]
        if low_precision:
            original_confidences = np.array(original_confidences, dtype=np.float16)
            confidences_over_backselect = np.array(confidences_over_backselect, dtype=np.float16)
            mask_order = np.array(mask_order, dtype=np.uint16)

        # Create directory containing file if it doesn't exist.
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        np.savez_compressed(
            filepath,
            original_confidences=original_confidences,
            target_class_idx=self.target_class_idx,
            confidences_over_backselect=confidences_over_backselect,
            mask_order=mask_order,
        )

    @staticmethod
    def from_file(filepath):
        """Instantiates BackselectResult from filepath.

        Args:
            filepath (str): Path to file.

        Returns:
            BackselectResult object.
        """
        data = np.load(filepath)
        res = BackselectResult(
            original_confidences=data['original_confidences'],
            target_class_idx=data['target_class_idx'],
            confidences_over_backselect=data['confidences_over_backselect'],
            mask_order=data['mask_order'],
        )
        return res




def run_gradient_backward_selection(images, model, remove_per_iter, max_iters=None,
                                    add_random_noise=True,
                                    random_noise_variance=1e-12,
                                    cuda=True):
    """Run Batched Gradient BackSelect."""
    assert len(images.shape) == 4  # Check for batch dimension.

    # Model to eval mode.
    model.eval()

    if cuda:
        images = images.cuda()

    # Initialize masks as all zeros.
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    masks = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3],
                        device=device, requires_grad=True)
    masks_history = torch.zeros(
        images.shape[0], 1, images.shape[2], images.shape[3], device=device,
        dtype=torch.int)
    # print(masks_history.shape)
    # print('masks.shape: ', masks.shape)

    # Compute initial predicted class.
    softmax = torch.nn.Softmax(dim=1)
    original_confidences = softmax(model(images))
    original_pred_confidences, original_pred_classes = original_confidences.max(axis=1)
    # print('original_pred_confidences: ', original_pred_confidences)
    # print('original_pred_classes: ', original_pred_classes)

    # Run backward selection.
    confidences_history = []

    if max_iters is None:
        max_iters = int(np.ceil(
            np.prod(masks.shape[1:]) / float(remove_per_iter)))
    # print('max_iters: ', max_iters)

    for i in range(max_iters):
        # Reset gradients.
        model.zero_grad()
        if masks.grad is not None:
            masks.grad.data.zero_()

        # Compute masked inputs.
        masked_images = (1 - masks) * images
        if cuda:
            masked_images = masked_images.cuda()

        # Compute confidences on masked images toward original predicted classes.
        confidences = softmax(model(masked_images))
        pred_confidences = confidences.gather(1, original_pred_classes.unsqueeze(1))
        # print(i, 'pred_confidences: ', pred_confidences.flatten().detach().cpu().numpy())

        # Compute gradients.
        torch.sum(pred_confidences).backward()
        assert masks.grad is not None
        grad_vals = masks.grad.detach()
        if add_random_noise:
            noise = (torch.randn(masks.shape) * (random_noise_variance**0.5))
            if cuda:
                noise = noise.cuda()
            grad_vals += noise

        # Find optimal pixels to mask, excluding previously masked values.
        not_yet_masked_idxs_tuple = (1 - masks).flatten(start_dim=1).nonzero(as_tuple=True)
        # We remove the same number of features per image per iteration, so all
        #   images have the same number of pixels remaining.
        grad_vals_not_yet_masked = grad_vals.flatten(start_dim=1)[not_yet_masked_idxs_tuple].reshape(masks.shape[0], -1)
        num_pixels_remaining = int((1 - masks[0]).sum())
        # print('num_pixels_remaining: ', num_pixels_remaining)
        _, to_mask_idxs_offset = torch.topk(
            grad_vals_not_yet_masked,
            min(remove_per_iter, num_pixels_remaining))
        # Remove offset from removing values for already masked pixels.
        not_yet_masked_idxs = not_yet_masked_idxs_tuple[1].reshape(
            masks.shape[0], -1)
        to_mask_idxs = not_yet_masked_idxs[
            torch.arange(masks.shape[0]).unsqueeze(1).expand(-1, to_mask_idxs_offset.shape[1]).flatten(),
            to_mask_idxs_offset.flatten(),
        ].reshape(masks.shape[0], -1)
        to_mask_idxs_mask = torch.zeros(masks.shape[0], masks.shape[2], masks.shape[3], device=device, dtype=torch.bool)
        to_mask_idxs_mask.view(masks.shape[0], -1)[
            torch.arange(masks.shape[0]).unsqueeze(1).expand(-1, to_mask_idxs_offset.shape[1]).flatten(),
            to_mask_idxs.flatten(),
        ] = 1
        to_mask_idxs_mask = to_mask_idxs_mask.unsqueeze(1)  # Add broadcast over channels dimension.
        assert bool(torch.all(to_mask_idxs_mask.sum(dim=(2,3)).flatten() == to_mask_idxs_offset.shape[1]))
        assert (to_mask_idxs_mask + masks).max() == 1

        # Update mask and history.
        with torch.no_grad():
            masks[to_mask_idxs_mask] = 1
            masks_history[to_mask_idxs_mask] = i

        confidences_history.append(confidences.detach().cpu().numpy())

    # Create BackselectResult objects.
    confidences_history = np.array(confidences_history)  # For slicing.
    to_return = []
    for i in range(images.shape[0]):
        to_return.append(BackselectResult(
            original_confidences=(
                original_confidences[i].detach().cpu().numpy()),
            target_class_idx=original_pred_classes[i].detach().cpu().numpy(),
            confidences_over_backselect=confidences_history[:, i, :],
            mask_order=masks_history[i].detach().cpu().numpy(),
        ))

    return to_return
