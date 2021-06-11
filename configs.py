# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pprint


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument('--verbose', type=str2bool, default='true')
    parser.add_argument('--pickle_path', type=str, default='./vix_future_preprocessed_july.pkl')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/') 
    parser.add_argument('--ckpt_path', type=str, default='test_01_seq_22_lay_1_hid1_12_ful1_2_ful2_0_epoch_150_gt1.pkl')

    # Model
    # parser.add_argument('--attention_mode', type=str2bool, default='true')
    parser.add_argument('--input_size', type=int, default=118)
    parser.add_argument('--num_layers', type=int, default=1)###
    parser.add_argument('--lstm_hidden_size', type=int, default=12)###180
    parser.add_argument('--full_hidden_size1', type=int, default=2)###60
    parser.add_argument('--full_hidden_size2', type=int, default=0)###10

    # Train
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--result_mode', type=int, default=1) 
    parser.add_argument('--n_epochs', type=int, default=300) ###
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--clip', type=float, default=5.0) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=22) ###
    parser.add_argument('--sequence_length', type=int, default=22) ###
    parser.add_argument('--test_ratio', type=float, default=0.1) 
    parser.add_argument('--drop_rate', type=float, default=0.0) ###

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
