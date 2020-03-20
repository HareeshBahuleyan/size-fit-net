import os
import json
import _jsonnet
import torch

from typing import Dict
from sklearn import metrics
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


def compute_metrics(target, pred_probs):
    """
    Computes metrics to report
    """
    pred_labels = pred_probs.argmax(-1)
    precision = metrics.precision_score(target, pred_labels, average="macro")
    recall = metrics.recall_score(target, pred_labels, average="macro")
    f1_score = metrics.f1_score(target, pred_labels, average="macro")
    accuracy = metrics.accuracy_score(target, pred_labels)
    auc = metrics.roc_auc_score(target, pred_probs, average="macro", multi_class="ovr")

    return precision, recall, f1_score, accuracy, auc
