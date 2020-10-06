"""Util for PyTorch model inference."""

import numpy as np
import os
import torch

import augmentations
from model import ensemble, model_util
from util import data_util, misc_util


def load_saved_model(saved_model_dir):
    # Check if model is an ensemble.
    ensemble_config_filepath = ensemble.get_ensemble_config_path(
        saved_model_dir)
    if os.path.exists(ensemble_config_filepath):
        print('Loading ensemble')
        model = ensemble.load_ensemble_from_config_file(
            ensemble_config_filepath)
        return model

    config_filepath = misc_util.get_config_path(saved_model_dir)
    config = misc_util.load_config(config_filepath)
    model_type = config['model']
    model_fn = model_util.get_model_from_name(model_type)
    dataset = config['dataset']
    num_classes = data_util.get_num_classes_for_dataset(dataset)
    model = model_fn(num_classes=num_classes)

    # Check if model wrapped with input droput.
    if 'input_dropout' in config and config['input_dropout']:
        print('Wrapping model with input dropout.')
        model = augmentations.ModelWithInputDropout(
            model,
            config['keep_prob'],
            num_samples=config['num_samples'],
        )

    checkpoint_path = misc_util.get_checkpoint_path(saved_model_dir)
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def predict(model, inputs, add_softmax=False):
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        if add_softmax:
            preds = torch.nn.functional.softmax(preds, dim=1)
    return preds


def predict_with_batching(model, inputs, batch_size, add_softmax=False):
    model.eval()
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))
    all_preds = []
    for batch_idx in range(num_batches):
        batch_start_i = batch_idx * batch_size
        batch_end_i = min(inputs.shape[0], (batch_idx + 1) * batch_size)
        assert batch_end_i > batch_start_i
        preds = predict(
            model,
            inputs[batch_start_i:batch_end_i],
            add_softmax=add_softmax,
        )
        all_preds.append(preds)
    all_preds = torch.cat(all_preds)
    return all_preds
