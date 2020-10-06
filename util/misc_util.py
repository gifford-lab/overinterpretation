"""Miscellaneous utils."""

import csv
import json
import os


CHECKPOINT_FILENAME = 'checkpoint.pt'
CONFIG_FILENAME = 'config.json'
LOG_FILENAME = 'log.csv'


class CSVLogger:
    def __init__(self, fieldnames, filepath):
        self.filepath = filepath
        self.csv_file = open(filepath, 'w')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def write_dict_to_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)


def write_args_to_json(args, filepath):
    args_dict = {}
    for arg in vars(args):
        args_dict[arg] = getattr(args, arg)
    write_dict_to_json(args_dict, filepath)


def get_log_path(root_dir):
    return os.path.join(root_dir, LOG_FILENAME)


def get_config_path(root_dir):
    return os.path.join(root_dir, CONFIG_FILENAME)


def get_checkpoint_path(root_dir):
    return os.path.join(root_dir, 'checkpoints', CHECKPOINT_FILENAME)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
        return config
