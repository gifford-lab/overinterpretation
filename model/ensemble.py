"""Ensemble model definitions."""

import json
import os
import torch

import inference_util


ENSEMBLE_CONFIG_FILENAME = 'ensemble_config.json'


class MeanEnsemble(torch.nn.Module):
    """Mean ensemble of models.

    Returns mean prediction over individual models.

    Args:
        models (list): List of models in the ensemble.
    """
    def __init__(self, models):
        super(MeanEnsemble, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        model_preds = torch.stack([model(x) for model in self.models])
        return torch.mean(model_preds, dim=0)

    def __len__(self):
        return len(self.models)


class MinEnsemble(torch.nn.Module):
    """Minimum ensemble of models.

    Returns minimum prediction over individual models.

    Args:
        models (list): List of models in the ensemble.
    """
    def __init__(self, models):
        super(MinEnsemble, self).__init__()
        torch.nn.ModuleList(models)

    def forward(self, x):
        model_preds = torch.stack([model(x) for model in self.models])
        return torch.min(model_preds, dim=0)[0]

    def __len__(self):
        return len(self.models)


_ENSEMBLE_TYPE_TO_CLASS = {
    'MeanEnsemble': MeanEnsemble,
}


def get_ensemble_config_path(root_dir):
    return os.path.join(root_dir, ENSEMBLE_CONFIG_FILENAME)


def load_ensemble_from_config(config):
    """Load ensemble from config.

    Args:
        config (dict): Dict containing `type` and `saved_model_dirs` keys.
    """
    ensemble_cls = _ENSEMBLE_TYPE_TO_CLASS[config['type']]

    # Load saved models.
    models = [
        inference_util.load_saved_model(d) for d in config['saved_model_dirs']]

    return ensemble_cls(models)


def load_ensemble_from_config_file(config_filepath):
    """Load ensemble from JSON config.

    Args:
        config_filepath (str): Path to config file (JSON). Should have fields
            `type` and `saved_model_dirs`.
    """
    with open(config_filepath) as f:
        config = json.load(f)

    return load_ensemble_from_config(config)
