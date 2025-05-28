import torch.optim as optim
import logging
from model import Graphormer
from pre_process import *
from utils import *


def Dismantling(graph, seed_nodes, threshold):
    A = graph.get_edgelist()
    G = nx.Graph(A)  # In case you graph is undirected
    TAS_NUM = 0
    TAS_CON = dict.fromkeys(seed_nodes)
    for node in seed_nodes:
        G.remove_node(node)
        if len(G) == 0:
            residual_largest_cc = 0
        else:
            residual_largest_cc = len(max(nx.connected_components(G), key=len))

        TAS_CON[node] = residual_largest_cc

    for node in seed_nodes:
        if TAS_CON[node] > threshold:
            TAS_NUM += 1
        else:
            break

    return TAS_NUM


def train_model(args, IO, graph_root, node_path_file, graph_type):
    early_stopping = EarlyStopping(logging, patience=args.patience, verbose=True)

    # 使用GPU or CPU
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))
    if args.gpu_index < 0:
        IO.cprint('Using CPU')
    else:
        if args.network_id == 0:
            IO.cprint('Using GPU: {}'.format(args.gpu_index))
        set_deterministic(args.seed)

    data, sparse_mask, attention_limit, batch_mask, threshold, num_nodes = preprocess(graph_root, args)
    node_path = get_node_path(node_path_file, data)

    data = data.to(device)

    # 加载模型及参数量统计
    model = Graphormer(args, data.count, numpy.max(node_path), data.ptr).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 打印模型信息
    if args.network_id == 0:
        IO.cprint(str(model))
        IO.cprint('Model Parameter: {}'.format(total_params))
        IO.cprint('Using AdamW')

    train_graph = LoadGraphData(graph_root)
    train_adj_matrices = sparse_mask.float().to(device)
    cos_mat = torch.zeros((data.x.shape[0], data.x.shape[0]), dtype=torch.int)

    best_TASpec = 1
    best_TAS = data.x.shape[0]

    IO.cprint(f'{graph_type} network:{args.network} id:{args.network_id} seed:{args.seed}')

    for epoch in range(args.epochs):
        #################
        #     Train     #
        #################
        model.train()  # 训练模式
        optimizer.zero_grad()

        attention_mask = sparse_mask.to(device) | cos_mat.to(device)
        score, feature_vec, cos_mat = model(data, node_path, attention_mask, attention_limit)
        score = score.squeeze(1)

        data.x = feature_vec

        _, TAS_indices = torch.topk(score, score.numel())

        TAS = Dismantling(train_graph, TAS_indices.cpu().numpy(), threshold)
        TAS_ratio = TAS / num_nodes

        loss = model.loss(score, train_adj_matrices.to(device), gamma=1, mean=False)

        if best_TASpec > TAS_ratio:
            best_TASpec = TAS_ratio
            best_epoch = epoch
            best_TAS = TAS
            IO.write(f"Best Epoch: {best_epoch} | Best TAS: {best_TASpec:.6f}")

        early_stopping(TAS_ratio, model, epoch)

        print(
            f"Train Epoch {epoch} | Loss: {loss:.4f} | TAS: {TAS}/{num_nodes}={TAS_ratio:.6f} ")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        loss.backward()
        optimizer.step()

    return best_TAS, best_TASpec
