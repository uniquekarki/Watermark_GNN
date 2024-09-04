import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader

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
def train(model, optimizer, criterion, data, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to validate the model
def validate(model, criterion, data, mask):
    model.eval()
    out = model(data)
    loss = criterion(out[mask], data.y[mask])
    return loss.item()

# Function to test the model
def test(model, data):
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    accuracy = int(correct.sum()) / int(data.test_mask.sum())
    return accuracy

# Function to run the training and evaluation process
def run_experiment(dataset):
    data = dataset[0]

    # Split the data into training, validation, and testing
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[:int(0.4 * data.num_nodes)] = True
    val_mask[int(0.4 * data.num_nodes):int(0.6 * data.num_nodes)] = True
    test_mask[int(0.6 * data.num_nodes):] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Initialize the model
    model = GraphSAGENet(dataset.num_features, 128, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 201):
        # Training step
        loss = train(model, optimizer, criterion, data, data.train_mask)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss:.4f}')

        # Validation step
        if epoch % 10 == 0:
            val_loss = validate(model, criterion, data, data.val_mask)
            print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}')

    # Testing the model
    accuracy = test(model, data)
    print(f'Test Accuracy: {accuracy:.4f}')

# Load and run the experiment for Cora dataset
print("Running experiment on Cora dataset...")
cora_dataset = load_dataset('Cora')
run_experiment(cora_dataset)

# Load and run the experiment for PubMed dataset
print("\nRunning experiment on PubMed dataset...")
pubmed_dataset = load_dataset('PubMed')
run_experiment(pubmed_dataset)
