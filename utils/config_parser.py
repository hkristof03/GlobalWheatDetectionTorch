import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description='Provide path to config file.'
    )
    parser.add_argument(
        '--pyaml',
        type=str,
        default='./configs/faster_rcnn_configs.yaml',
        help='Path to yaml config relative to base dir (project dir).'
    )
    return parser.parse_args()

def parse_yaml(path_yaml):
    with open(path_yaml, 'r') as f:
        configs = yaml.load(f.read(), Loader=yaml.Loader)
    return configs
