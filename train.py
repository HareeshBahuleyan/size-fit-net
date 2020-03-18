import argparse

from collections import OrderedDict
from utils import load_config_from_json
from modcloth import ModCloth

def main(args):
    data_config = load_config_from_json(args.data_config_path)
    model_config = load_config_from_json(args.model_config_path)

    splits = ['train', 'valid', 'test']

    datasets = OrderedDict()
    datasets['valid'] = ModCloth(data_config, split='valid')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str, default='configs/data.jsonnet')
    parser.add_argument('--model_config_path', type=str, default='configs/model.jsonnet')

    args = parser.parse_args()
    main(args)