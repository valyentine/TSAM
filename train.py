from train_model import train_model
import os
from parameter import parse_args, IOStream, table_printer
import time
import random

graph_type = {
    'BA_1000_4': 'synthetic',
    'ER_1000_6': 'synthetic',
    'WS_1000_8_0.8': 'synthetic',
    'PLC_1000_5_0.1': 'synthetic',
    'AirTraffic': 'realworld',
    'Chicago': 'realworld',
    'Euroroads': 'realworld',
    'Figeys': 'realworld',
    'FilmTrust': 'realworld',
    'Gnutella': 'realworld',
    'LastFM': 'realworld',
    'PPI': 'realworld',
    'RoviraVirgili': 'realworld',
    'Genefusion': 'realworld',
    'Vidal': 'realworld',
}


def exp_init():
    """实验初始化"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')


def train_synthetic(args, IO, graph_type, num_instances=20):
    """训练合成数据"""
    best_TAS_list = []

    for num_instances in range(num_instances):
        args.network_id = num_instances
        graph_id = str(args.network_id)
        graph_root = os.path.join(root, f'{graph_id}.edge')
        node_path_file = os.path.join(root, 'node_path', f'node_path_{graph_id}.pkl')
        best_TAS, best_TAS_perc = train_model(args, IO, graph_root, node_path_file, graph_type)
        best_TAS_list.append(best_TAS)
    IO.cprint(f'Best TAS of every graph: {best_TAS_list}')
    IO.cprint(f'Mean Best TAS: {sum(best_TAS_list) / len(best_TAS_list)}')


def train_realworld(args, IO, graph_type):
    """训练真实数据"""
    graph_root = os.path.join(root, f'{args.network}.txt')
    node_path_file = os.path.join(root, 'node_path', f'node_path.pkl')
    best_TAS, best_TAS_perc = train_model(args, IO, graph_root, node_path_file, graph_type)
    IO.cprint(f'Best TAS: {best_TAS} | Best TAS Percentage: {best_TAS_perc}')


if __name__ == '__main__':
    args = parse_args()
    if args.use_random_seed:
        args.seed = random.randint(0, 2**32 - 1)
    exp_init()
    
    if args.network in graph_type:
        root = os.path.join('data', graph_type[args.network], args.network)
        DATE = time.strftime('%m-%d', time.localtime())
        TIME = time.strftime('%H.%M.%S', time.localtime())
        IO = IOStream('outputs/' + f'{args.network}_{DATE}_{TIME}.log')
        IO.cprint(str(table_printer(args)))

        if graph_type[args.network] == 'synthetic':
            train_synthetic(args, IO, graph_type[args.network])

        if graph_type[args.network] == 'realworld':
            train_realworld(args, IO, graph_type[args.network])

        IO.close()
    else:
        print('ERROR:Invalid network name!')

