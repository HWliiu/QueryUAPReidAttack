import argparse

import torch
import accelerate

import config

from tools.launch import launch
from utils import setup_logger


def main(args):
    cfg = config.get_default_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    accelerator = accelerate.Accelerator(fp16=False)

    logger = setup_logger(
        name='reidattack',
        distributed_rank=accelerator.local_process_index)

    # seed = 1000
    # accelerate.utils.set_seed(seed)
    # logger.info(f'Set all seed {seed}')

    if args.config_file:
        logger.info(f'Loaded configuration file {args.config_file}')

    launch(cfg, accelerator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()
    main(args)
