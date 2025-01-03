import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader
from watermark import generate_er_graph, combine_graphs

# Function to load the dataset
def load_dataset(name):
    dataset = Planetoid(root=f'/tmp/{name}', name=name, transform=NormalizeFeatures())
    return dataset

# Define the GraphSAGE model
class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')  # First layer with 128 neurons
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')  # Second layer with 128 neurons

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second layer with softmax output
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Softmax for output

# Function to train the model
def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to validate the model
def validate(model, criterion, data):
    model.eval()
    out = model(data)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss.item()

def test_on_er_graph(model, er_data):
    model.eval()
    # Assuming all nodes in ER graph are test nodes
    er_data.test_mask = torch.ones(er_data.num_nodes, dtype=torch.bool)
    
    with torch.no_grad():
        out = model(er_data)
        pred = out.argmax(dim=1)
        correct = (pred[er_data.test_mask] == er_data.y[er_data.test_mask]).sum()
        accuracy = int(correct) / int(er_data.test_mask.sum())
    
    return accuracy

# Function to test the model
def test(model, data):
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    accuracy = int(correct.sum()) / int(data.test_mask.sum())
    return accuracy

def datasplit(data, watermark = False, er_data = None):
    # Split the data into training, validation, and testing
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    if watermark:
        graph_nodes = er_data.num_nodes #This is used in masks as we need to split the original dataset and then attach the graph nodes to training

        train_mask[:int(0.4 * data.num_nodes) + graph_nodes] = True
        val_mask[int(0.4 * data.num_nodes) + graph_nodes:int(0.6 * data.num_nodes) + graph_nodes] = True
        test_mask[int(0.6 * data.num_nodes) + graph_nodes:] = True
    else:
        train_mask[:int(0.4 * data.num_nodes)] = True
        val_mask[int(0.4 * data.num_nodes):int(0.6 * data.num_nodes)] = True
        test_mask[int(0.6 * data.num_nodes):] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

# Function to run the training and evaluation process
def run_experiment(dataset, dataset_name, er_data = None, watermark = False):
    if watermark:
        data = combine_graphs(er_data, dataset[0])
        data = datasplit(data, watermark, er_data)
    else:
        data = datasplit(dataset[0])

    # Initialize the model
    model = GraphSAGENet(dataset.num_features, 128, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 201):
        # Training step
        loss = train(model, optimizer, criterion, data)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss:.4f}')

        # Validation step
        if epoch % 10 == 0:
            val_loss = validate(model, criterion, data)
            print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}')

    if watermark:
        model_save_path = f"../models/{dataset_name}_watermarked_gnn_model.pth"
    else:
        model_save_path = f"../models/{dataset_name}_base_gnn_model.pth"
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model

def fine_tuning(model, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 201):
        # Training step
        loss = train(model, optimizer, criterion, dataset)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss:.4f}')

        # Validation step
        if epoch % 10 == 0:
            val_loss = validate(model, criterion, dataset)
            print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}')
    
    return model

def print_accuracy(model, data, er_data, model_name):
    # Test on ER graph only
    er_graph_accuracy = test_on_er_graph(model, er_data)
    print(f'{model_name} ER Graph Accuracy: {er_graph_accuracy:.4f}')

    # Testing the model
    accuracy = test(model, data)
    print(f'{model_name} Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":

    pr = 0.3
    pg = 0.2
    n = 1
    # CORA DATASET
    # Load and run the experiment for Cora dataset
    print("Running experiment on Cora dataset...")
    cora_dataset = load_dataset('Cora')

    # Extract feature size and number of classes
    feature_size = cora_dataset.num_node_features
    num_classes = cora_dataset.num_classes
    cora_er_data = generate_er_graph(n=n, p=pg, seed=191, feature_size=feature_size, num_classes=num_classes, pr = pr)
    
    cora_unmarked_model = run_experiment(cora_dataset, "cora")
    print_accuracy(cora_unmarked_model, cora_dataset, cora_er_data, "cora")

    cora_marked_model = run_experiment(cora_dataset, "cora", cora_er_data, True)
    print_accuracy(cora_marked_model, cora_dataset, cora_er_data, "cora")

    # Unlearning prelim study
    unlearn_dataset = datasplit(cora_dataset[0])
    unlearned_model = fine_tuning(cora_marked_model, unlearn_dataset)
    print_accuracy(unlearned_model, unlearn_dataset, cora_er_data, "cora")

    #PUBMED DATA
    # Load and run the experiment for PubMed dataset
    print("\nRunning experiment on PubMed dataset...")
    pubmed_dataset = load_dataset('PubMed')

    # Extract feature size and number of classes
    feature_size = pubmed_dataset.num_node_features
    num_classes = pubmed_dataset.num_classes
    pubmed_er_data = generate_er_graph(n=n, p=pg, seed=191, feature_size=feature_size, num_classes=num_classes, pr = pr)

    pubmed_unmarked_model = run_experiment(pubmed_dataset, "pubmed")
    print_accuracy(pubmed_unmarked_model, pubmed_dataset, pubmed_er_data, "pubmed")

    pubmed_marked_model = run_experiment(pubmed_dataset, "pubmed", pubmed_er_data, True)
    print_accuracy(pubmed_marked_model, pubmed_dataset, pubmed_er_data, "pubmed")

    # Unlearning prelim study
    unlearn_dataset = datasplit(pubmed_dataset[0])
    unlearned_model = fine_tuning(pubmed_marked_model, unlearn_dataset)
    print_accuracy(unlearned_model, unlearn_dataset, pubmed_er_data, "pubmed")