import os
import json
import _jsonnet
import torch

from typing import Dict
from torch.autograd import Variable


def load_config_from_json(config_file: str) -> Dict:
    # load configuration
    if not os.path.isfile(config_file):
        raise ValueError("given configuration file doesn't exist")
    with open(config_file, "r") as fio:
        config = fio.read()
        config = json.loads(_jsonnet.evaluate_snippet("", config))
    return config


def to_var(x, volatile=False):
    # To convert tensors to CUDA tensors if GPU is available
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
