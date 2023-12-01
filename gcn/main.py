from torch_geometric.datasets import KarateClub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.w = nn.Linear(in_features=in_features, out_features=out_features, bias=False) 

    def forward(self, A, X):
        h = self.w(X)
        h = A @ h 
        return h
    

class GCNModel(nn.Module):

    def __init__(self, in_features, hidden_dim, num_classes):
        super(GCNModel, self).__init__()
        self.l1 = GCNLayer(in_features, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, A, X):
        H = self.l1(A, X)
        H = F.relu(H)
        H = self.l2(A, H)
        return H 
    

if __name__ == '__main__':
    dataset = KarateClub()
    data = dataset[0]
    x = data.x 
    y = data.y
    train_mask = data.train_mask
    edge_index = data.edge_index
    num_nodes = data.y.shape[0]
    num_edges = edge_index.shape[1]

    A = torch.sparse_coo_tensor(edge_index, torch.ones(num_edges), (num_nodes, num_nodes))
    A = A.to_dense()
    A += torch.eye(num_nodes)
    D_inv = torch.diag(1 / A.sum(0))
    A = A @ D_inv

    model = GCNModel(in_features=num_nodes, hidden_dim=8, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    valid_losses = []
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(A, x)
        train_loss = criterion(output[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
        
        with torch.no_grad():
            valid_loss = criterion(output[~train_mask], y[~train_mask])
            valid_losses.append(valid_loss)


    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.show()