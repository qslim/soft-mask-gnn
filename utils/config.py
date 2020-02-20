import json
from easydict import EasyDict


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = EasyDict(config_dict)

    return config


def process_config(args):
    config = get_config_from_json(args.config)
    return config


if __name__ == '__main__':
    config = process_config('../configs/MUTAG.json')
    print(config)
