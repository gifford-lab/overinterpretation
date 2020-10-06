"""Util for analysis of SIS/backselect data."""

import collections
import numpy as np
import os
import torch

import inference_util
import sis_util
from sufficient_input_subsets import sis


# Function to sort filenames by image index in path.
SR_SORT = lambda s: int(os.path.basename(s).split('_')[-1].split('.')[0])


LoadSISResults = collections.namedtuple(
    'LoadSISResults',
    [
        'sis_results',
        'sis_image_idxs',
        'sis_pred_class',
        'sis_is_correct_class',
        'sis_masked_images',
        'original_confidences',
    ],
)


def backselect_mask_from_sis_result(sis_result, features_to_keep):
    backselect_mask = np.zeros(sis_result.mask.shape, dtype=bool)
    backselect_mask[sis._transform_index_array_into_indexer(
        sis_result.ordering_over_entire_backselect[-features_to_keep:])] = True
    return backselect_mask


def find_sis_from_backselect_result(sis_result, threshold):
    # Assumes SIS exists (initial prediction >= threshold).
    backselect_stack = list(zip(
        sis_result.ordering_over_entire_backselect,
        sis_result.values_over_entire_backselect,
    ))

    sis_idxs = sis._find_sis_from_backselect(backselect_stack, threshold)

    mask = ~(sis.make_empty_boolean_mask(sis_result.mask.shape))
    mask[sis._transform_index_array_into_indexer(sis_idxs)] = True

    new_sis_result = sis.SISResult(
      sis=np.array(sis_idxs, dtype=np.int_),
      ordering_over_entire_backselect=np.array(
          sis_result.ordering_over_entire_backselect, dtype=np.int_),
      values_over_entire_backselect=np.array(
          sis_result.values_over_entire_backselect, dtype=np.float_),
      mask=mask,
    )

    return new_sis_result


def load_sis_results(dataset, dataset_name, model, sis_results_dir,
                     fully_masked_image, sis_threshold, max_num=None):
    """Load data and create masks and masked images."""
    sis_results = []
    sis_image_idxs = []
    sis_pred_class = []
    sis_is_correct_class = []
    sis_masked_images = []
    original_confidences = []

    num_images = len(dataset)
    if max_num:
        num_images = min(max_num, len(dataset))
    for i in range(num_images):
        image, label = dataset[i]

        # Check if original prediction >= threshold.
        original_preds = inference_util.predict(
            model, image.unsqueeze(0).cuda(), add_softmax=True)
        original_confidence = float(original_preds.max())
        original_label = int(original_preds.argmax())
        if original_confidence < sis_threshold:
            continue  # No SIS exists.

        # Compute SIS from backselect data.
        sis_file = os.path.join(
            sis_results_dir, '%s_%d.npz' % (dataset_name, i))

        backselect_sr = sis_util.load_sis_result(sis_file)
        sis_result = find_sis_from_backselect_result(
            backselect_sr, sis_threshold)
        sis_masked_image = sis.produce_masked_inputs(
            image.numpy(), fully_masked_image, [sis_result.mask])[0]

        sis_results.append(sis_result)
        sis_image_idxs.append(i)
        sis_pred_class.append(original_label)
        sis_is_correct_class.append((original_label == label))
        sis_masked_images.append(sis_masked_image)
        original_confidences.append(original_confidence)

    sis_image_idxs = np.array(sis_image_idxs)
    sis_pred_class = np.array(sis_pred_class)
    sis_is_correct_class = np.array(sis_is_correct_class)
    sis_masked_images = np.array(sis_masked_images)
    original_confidences = np.array(original_confidences)

    return LoadSISResults(
        sis_results=sis_results,
        sis_image_idxs=sis_image_idxs,
        sis_pred_class=sis_pred_class,
        sis_is_correct_class=sis_is_correct_class,
        sis_masked_images=sis_masked_images,
        original_confidences=original_confidences,
    )


def load_backselect_subsets(dataset, dataset_name, pixels_to_keep,
                            sis_results_dir, fully_masked_image, max_num=None):
    bs_masks = []
    bs_masked_images = []

    num_images = len(dataset)
    if max_num:
        num_images = min(max_num, len(dataset))
    for i in range(num_images):
        image, _ = dataset[i]
        sis_file = os.path.join(
            sis_results_dir, '%s_%d.npz' % (dataset_name, i))
        sr = sis_util.load_sis_result(sis_file)
        bs_mask = backselect_mask_from_sis_result(sr, pixels_to_keep)
        img_masked = sis.produce_masked_inputs(
            image.numpy(), fully_masked_image, [bs_mask])[0]
        bs_masks.append(bs_mask)
        bs_masked_images.append(img_masked)

    bs_masks = np.array(bs_masks)
    bs_masked_images = np.array(bs_masked_images)

    return bs_masks, bs_masked_images
