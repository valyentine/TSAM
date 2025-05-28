import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from complete import detect


def get_graph(file_path):
    edges = []
    nodes = set()
    with open(file_path, 'r') as f:
        for line in f:
            n1, n2 = map(int, line.strip().split())
            edges.append((n1, n2))
            edges.append((n2, n1))
            nodes.update([n1, n2])

    edges = sorted(edges)
    num_nodes = len(nodes)

    return edges, nodes, num_nodes


def get_degress(edges, nodes):
    node_degrees = {node: 0 for node in nodes}
    for n1, n2 in edges:
        node_degrees[n1] += 1
        node_degrees[n2] += 1

    return node_degrees


def get_limit(node_degrees, min_sample, max_sample):
    AttentionLimit = []
    for degrees in node_degrees:
        node_limit_min = max(node_degrees[degrees] / 2, min_sample)
        node_limit_max = min(node_limit_min, max_sample)
        AttentionLimit.append(int(node_limit_max))

    return AttentionLimit


def filter_and_reindex(edges, threshold, components):
    filtered_nodes = set()
    filtered_components = []

    # Filter components based on the threshold
    for component in components:
        if len(component) < threshold:
            break
        filtered_nodes.update(component)
        filtered_components.append(component)

    # Create a mapping from old to new node indices
    old_to_new = {old: new for new, old in enumerate(sorted(filtered_nodes))}
    new_nodes = set(old_to_new.values())

    # Update edges with new node indices
    new_edges = []
    for n1, n2 in edges:
        if n1 in old_to_new and n2 in old_to_new:
            new_edges.append((old_to_new[n1], old_to_new[n2]))

    # Reindex components with new node indices
    reindexed_components = []
    for component in filtered_components:
        reindexed_components.append({old_to_new[node] for node in component})

    return new_edges, new_nodes, reindexed_components


def preprocess(root, args):
    print("using original graph")
    # 读取文件并获取节点和边信息
    file_path = root
    minimum_sample = args.minimum_sample
    maximum_sample = args.maximum_sample
    input_dim = args.input_dim

    edges, nodes, num_nodes = get_graph(file_path)
    threshold = args.threshold * num_nodes
    sum_nodes = num_nodes

    components = detect(file_path)
    components_count = len(components)
    if components_count > 1:
        edges, nodes, components = filter_and_reindex(edges, threshold, components)
        num_nodes, components_count = len(nodes), len(components)

    node_degrees = get_degress(edges, nodes)
    AttentionLimit = get_limit(node_degrees, minimum_sample, maximum_sample)

    # 初始化节点特征矩阵 x
    degrees_tensor = torch.tensor([node_degrees[node] for node in sorted(nodes)], dtype=torch.float).view(-1, 1)

    unique_degrees, inverse_indices = torch.unique(degrees_tensor, return_inverse=True)
    rank = inverse_indices.squeeze(1)
    count = unique_degrees.size(0)

    initTensor = torch.nn.init.xavier_uniform_(torch.empty(num_nodes, input_dim))
    x = initTensor

    # 构建边索引矩阵 edge_index
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    neighbor_mask = torch.zeros(size=(num_nodes, num_nodes), dtype=torch.int)
    for i in range(edge_index.size(1)):
        neighbor_mask[edge_index[0, i], edge_index[1, i]] = 1

    # 初始化 batch, ptr
    batch = torch.zeros(num_nodes, dtype=torch.long)

    batch_mask = torch.zeros(size=(x.shape[0], x.shape[0]), dtype=torch.int)

    if components_count == 1:
        ptr = torch.tensor([0], dtype=torch.long)
        batch_mask += 1
        data = Data(x=x, edge_index=edge_index, batch=batch, ptr=ptr, count=count, rank=rank)

    else:
        ptr_list = [0]
        len_count = 0
        for i, component in enumerate(components):
            len_count += len(component)
            ptr_list.append(len_count)
            for j in component:
                batch[j] += i
        ptr = torch.tensor(ptr_list, dtype=torch.long)
        for i in range(len(ptr) - 1):
            batch_mask[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
        data_list = [
            Data(x=x, edge_index=edge_index, batch=batch, ptr=ptr, count=count, rank=rank)]
        data = Batch.from_data_list(data_list)
        data.ptr = ptr
        data.batch = batch

    return data, neighbor_mask, AttentionLimit, batch_mask, threshold, sum_nodes
