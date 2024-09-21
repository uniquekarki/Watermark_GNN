import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

def generate_er_graph(n, p, seed, feature_size, num_classes, pr):
    # Generate ER graph
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    
    # Initialize node features as all zeros
    features = torch.zeros((n, feature_size))
    
    # Set a proportion of elements in the feature matrix to 1 based on pr
    num_ones = int(pr * feature_size)  # Number of ones per node's feature vector
    for i in range(n):
        ones_indices = torch.randperm(feature_size)[:num_ones]  # Randomly pick indices to set to 1
        features[i, ones_indices] = 1.0

    # Set node features and random labels
    nx.set_node_attributes(G, {i: features[i].tolist() for i in G.nodes()}, 'x')  # Features as 0/1 vectors
    nx.set_node_attributes(G, {i: torch.randint(0, num_classes, (1,)).item() for i in G.nodes()}, 'y')  # Random labels
    
    return from_networkx(G)

# Example of augmenting Cora dataset with ER graph
def combine_graphs(graph1, graph2):
    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index + graph1.num_nodes], dim=1)
    combined_y = torch.cat([graph1.y, graph2.y], dim=0)

    return Data(x=combined_x, edge_index=combined_edge_index, y=combined_y)