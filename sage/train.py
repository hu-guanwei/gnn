import logging
from itertools import accumulate
from collections import defaultdict
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from GraphSage import GraphSage
from torch_geometric.datasets import Planetoid
from node_sampling import multi_hop_sampling
from tqdm import tqdm

INPUT_DIM = 1433
HIDDEN_DIM = [128, 7]
NUM_NEIGHBORS_LIST = [5, 10]
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)

BTACH_SIZE = 1
EPOCHS = 1000
NUM_BATCH_PER_EPOCH = 20
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def edge_index_to_adjacency_dict(edge_index):
    adjacency_dict = defaultdict(set)

    for i in range(edge_index.size(1)):
        src_node = edge_index[0][i].item()
        dst_node = edge_index[1][i].item()
        adjacency_dict[src_node].add(dst_node)

    return dict(adjacency_dict)


dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x = data.x / data.x.sum(1, keepdims=True)
y = data.y
adj_map = edge_index_to_adjacency_dict(data.edge_index)

train_index = np.arange(0, len(x))[data.train_mask]
test_index = np.arange(0, len(x))[data.test_mask]

num_nodes_per_layer = list(accumulate([BTACH_SIZE] + NUM_NEIGHBORS_LIST, lambda x, y: x * y))
model = GraphSage(input_dim=INPUT_DIM, hidden_dim_list=HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)


def train():
    model.train()
    for e in tqdm(range(EPOCHS)):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            batch_src_label = y[batch_src_index].long().to(DEVICE)

            batch_sampling_result: List[List[int]] = multi_hop_sampling(batch_src_index.tolist(),
                                                                        NUM_NEIGHBORS_LIST,
                                                                        adj_map)
            batch_sampling_x: List[torch.Tensor] = []
            for sample_this_layer in batch_sampling_result:
                batch_sampling_x.append(x[sample_this_layer])

            batch_train_logits = model(batch_sampling_x, num_nodes_per_layer)
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multi_hop_sampling(test_index, NUM_NEIGHBORS_LIST, adj_map)

        test_x: List[torch.Tensor] = []
        for sample_this_layer in test_sampling_result:
            test_x.append(x[sample_this_layer])

        test_logits = model(test_x,
                            num_nodes_per_layer=list(accumulate([test_index.size] + NUM_NEIGHBORS_LIST, lambda x, y: x * y)))
        test_label = y[test_index].long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuracy = torch.eq(predict_y, test_label).float().mean().item()
        logging.info(f"Test Accuracy: {accuracy}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', filename='train.log', filemode='w')
    train()
    test()
