import os
import json
import _jsonnet
from typing import Dict

def load_config_from_json(config_file: str) -> Dict:
    # load configuration
    if not os.path.isfile(config_file):
        raise ValueError("given configuration file doesn't exist")
    with open(config_file, "r") as fio:
        config = fio.read()
        config = json.loads(_jsonnet.evaluate_snippet("", config))
    return config