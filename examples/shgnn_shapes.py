import sys

sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import GeometricShapes
from torch_geometric.loader import DataLoader

from mnca.models import MLP, SHGNN

train_dataset = GeometricShapes(root="data/GeometricShapes", train=True)
train_dataset.s = train_dataset.pos
train_dataset.transform = T.Compose(
    [T.SamplePoints(num=256), T.KNNGraph(k=6), T.Spherical()]
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = GeometricShapes(root="data/GeometricShapes", train=False)
test_dataset.s = test_dataset.pos
test_dataset.transform = T.Compose(
    [T.SamplePoints(num=256), T.KNNGraph(k=6), T.Spherical()]
)
test_loader = DataLoader(test_dataset, batch_size=32)

mlp = MLP(3, 1, 1, 8)
shgnn = SHGNN(mlp)

optimizer = optim.Adam(shgnn.parameters(), lr=0.01)
criterion = nn.MSELoss()


def train():
    shgnn.train()
    total_loss = 0.0

    for data in train_loader:
        optimizer.zero_grad()
        out = shgnn(data.pos, None, data.edge_attr[:, 1:], data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss


for epoch in range(1, 51):
    loss = train()
    # test_acc = test()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")  # , Test Acc: {test_acc:.4f}")
