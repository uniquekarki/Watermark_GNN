import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

# Generate an Erdos-Renyi Graph using networkx
def generate_er_graph(n, p, seed, feature_size, num_classes):
    # Generate an Erdős-Rényi graph
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    
    # Assign random features to nodes (size: feature_size)
    features = torch.randn(n, feature_size).tolist()
    nx.set_node_attributes(G, {i: features[i] for i in G.nodes()}, 'x')
    
    # Assign random labels to nodes (num_classes)
    labels = torch.randint(0, num_classes, (n,)).tolist()
    nx.set_node_attributes(G, {i: labels[i] for i in G.nodes()}, 'y')
    
    # Convert to PyTorch Geometric Data object
    data = from_networkx(G)
    
    # Convert node attributes to tensors
    data.x = torch.tensor([G.nodes[i]['x'] for i in G.nodes()], dtype=torch.float)
    data.y = torch.tensor([G.nodes[i]['y'] for i in G.nodes()], dtype=torch.long)
    
    return data

# Example of augmenting Cora dataset with ER graph
def combine_graphs(graph1, graph2):
    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index + graph1.num_nodes], dim=1)
    combined_y = torch.cat([graph1.y, graph2.y], dim=0)

    return Data(x=combined_x, edge_index=combined_edge_index, y=combined_y)