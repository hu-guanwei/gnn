import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List


class NeighborAggregator(nn.Module):

    def __init__(self, input_dim, output_dim, aggr_method='mean'):
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.reset_parameters()
        self.aggr_method = aggr_method

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, neighbors_x):
        """
        :param neighbors_x: shape (num_src_nodes, num_neighbors, input_dim), sample num_neighbors for each source nodes
        :return:
        """
        if self.aggr_method == 'mean':
            neighbors_x_aggr = neighbors_x.mean(dim=1)  # mean over different neighbors
        else:
            raise NotImplementedError

        neighbors_hidden = torch.matmul(neighbors_x_aggr, self.weight)
        return neighbors_hidden


class SageLayer(nn.Module):

    def __init__(self, input_dim, output_dim,
                 activation=F.relu,
                 aggr_neighbor_method='mean',
                 aggr_hidden_method='sum'):

        super(SageLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.neighbor_aggregator = NeighborAggregator(input_dim, output_dim,
                                                      aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, src_x, neighbors_x):
        neighbors_hidden = self.neighbor_aggregator(neighbors_x)
        self_hidden = torch.matmul(src_x, self.weight)
        if self.aggr_hidden_method == 'sum':
            hidden = self_hidden + neighbors_hidden
        else:
            raise NotImplementedError

        if self.activation:
            return self.activation(hidden)
        return hidden


class GraphSage(nn.Module):

    def __init__(self, input_dim, hidden_dim_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list
        self.num_hidden_layers = len(hidden_dim_list)

        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SageLayer(input_dim, hidden_dim_list[0]))

        if self.num_hidden_layers > 1:
            for i in range(len(hidden_dim_list) - 2):
                self.sage_layers.append(SageLayer(hidden_dim_list[i], hidden_dim_list[i+1]))
            self.sage_layers.append(SageLayer(hidden_dim_list[-2], hidden_dim_list[-1]))

    def forward(self, x_list: List[torch.Tensor], num_nodes_per_layer: List[int]) -> torch.Tensor:
        """
        :param x_list: [source_nodes_x, 1-hop neighbors_x, 2-hop neighbors_x, ...]
        :param num_nodes_per_layer: number of sampled nodes in each layer
        """
        assert self.num_hidden_layers + 1 == len(x_list)

        hidden_list = x_list
        for i in range(self.num_hidden_layers):
            next_layer_hidden_list = []
            sage_layer = self.sage_layers[i]

            for hop in range(len(hidden_list) - 1):
                src_num = num_nodes_per_layer[hop]
                src_x = hidden_list[hop].view(src_num, -1)
                # size: (num_of_src_nodes, hidden_dim)
                neighbors_x = hidden_list[hop + 1].view(src_x.size(0), -1, src_x.size(1))
                # size: (num_of_src_nodes, num_of_sampled_neighbors_per_src_nodes, hidden_dim)

                h = sage_layer(src_x, neighbors_x)
                next_layer_hidden_list.append(h)

            hidden_list = next_layer_hidden_list

        return hidden_list[0]


if __name__ == '__main__':
    num_src_nodes = 3
    num_neighbors = 2
    input_dim = 4
    output_dim = 5

    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\n%(message)s')

    # test NeighborAggregator
    neighbor_aggregator = NeighborAggregator(input_dim=input_dim, output_dim=output_dim, aggr_method='mean')
    neighbor_x = torch.randn((num_src_nodes, num_neighbors, input_dim))
    logging.debug(neighbor_x)
    logging.debug(neighbor_aggregator(neighbor_x))

    # test SageLayer
    sage_layer = SageLayer(input_dim=input_dim, output_dim=output_dim)
    src_x = torch.randn((num_src_nodes, input_dim))
    z = sage_layer(src_x, neighbor_x)
    logging.debug(z)

    # test GraphSage
    sage_model = GraphSage(input_dim=input_dim,
                           hidden_dim_list=[output_dim, output_dim])
    two_hop_neighbor_x = torch.randn((num_src_nodes * num_neighbors, num_neighbors, input_dim))
    logging.debug(src_x.shape)
    logging.debug(neighbor_x.shape)
    logging.debug(two_hop_neighbor_x.shape)
    z = sage_model([src_x, neighbor_x, two_hop_neighbor_x], num_nodes_per_layer=[num_src_nodes,
                                                                                 num_neighbors * num_src_nodes,
                                                                                 num_neighbors * num_neighbors * num_src_nodes])
    logging.debug(z)

