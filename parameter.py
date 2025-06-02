#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:57
"""
import argparse
from texttable import Texttable


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', type=str, default='RoviraVirgili', help='Name of the network')
    parser.add_argument('--network_id', type=int, default=0, help='ID of the network')
    parser.add_argument('--seed', type=int, default=16 , help='Random seed of the experiment')
    parser.add_argument('--patience', type=int, default=60, help='Patience of early stopping')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold of early stopping')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate of AdamW')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay of AdamW')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--minimum_sample', type=int, default=3, help='Maximum number of node attention')
    parser.add_argument('--maximum_sample', type=int, default=10, help='minimum number of node attention')
    parser.add_argument('--input_dim', type=int, default=8, help='Input dimension of node features')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimensions of node features')
    parser.add_argument('--output_dim', type=int, default=1, help='Number of output node features')
    parser.add_argument('--use_random_seed', type=bool, default=True, help='Use random seed')

    args = parser.parse_args()

    return args


class IOStream:
    """训练日志文件"""
    def __init__(self, path):
        self.file = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.file.write(text + '\n')
        self.file.flush()

    def write(self, text):
        self.file.write(text + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


def table_printer(args):
    """绘制参数表格"""
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.set_cols_dtype(['t', 't'])
    rows = [["Parameter", "Value"]]
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) # 下划线替换成空格，首字母大写
    table.add_rows(rows)
    return table.draw()
