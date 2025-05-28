import networkx as nx
import os

def get_connected_components(graph):
    G = nx.Graph(graph.get_edgelist())
    components = list(nx.connected_components(G))
    return components

def number_mapping(components):
    num_nodes = 0
    for component in components:
        num_nodes += len(component)

    number_mapping = [0 for _ in range(num_nodes)]
    nodeCtn = 0
    for component in components:
        for idx in component:
            number_mapping[idx] = nodeCtn
            nodeCtn += 1

    return number_mapping

class CustomGraph:
    def __init__(self):
        self.edges = []

    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))

    def get_edgelist(self):
        return self.edges

def detect(root):
    graph = CustomGraph()
    with open(root, 'r') as f:
        lines = f.readlines()

    for line in lines:
        node1, node2 = map(int, line.split())
        graph.add_edge(node1, node2)

    components = get_connected_components(graph)
    components.sort(key=len, reverse=True)  # Sort components by size in descending order
    # for i, component in enumerate(components):
    #     print(f'Component {i + 1}: {component}')
    return components

def rewrite(root, output, Mapping=None):
    input_file = root
    output_file = output
    nodeMapping = Mapping

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            numbers = list(map(int, line.split()))
            transformed_numbers = [nodeMapping[number] for number in numbers]
            outfile.write(' '.join(map(str, transformed_numbers)) + '\n')

if __name__ == '__main__':
    network = 'your_network'
    input_root = os.path.join('data', 'realworld', network, f'{network}.txt')
    output_root = os.path.join('data', 'realworld', network, f'{network}_new.txt')
    components = detect(input_root)
    MappingNodes = number_mapping(components)
    rewrite(input_root, output_root, MappingNodes)
    os.remove(input_root)
    os.rename(output_root, input_root)
