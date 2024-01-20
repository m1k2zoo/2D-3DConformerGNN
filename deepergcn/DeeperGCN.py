import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_scatter import scatter


class DeeperGCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, edge_attr_dim, hidden_channels, num_layers, 
                 reduce='sum'):
        super().__init__()

        self.node_encoder = Linear(input_dim, hidden_channels)
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, output_dim)
        self.reduce = reduce


    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        embeddings = F.dropout(x, p=0.1, training=self.training)

        single_values_features = self.lin(embeddings)
        single_values_per_batch = scatter(single_values_features.T, batch, reduce=self.reduce)

        node_embeddings = embeddings
        global_embeddings = scatter(embeddings.T, batch, reduce=self.reduce).T

        return single_values_per_batch, global_embeddings, node_embeddings

    # def extract_embedding(self, x, edge_index, edge_attr, batch):
    #     x = self.node_encoder(x)
    #     edge_attr = self.edge_encoder(edge_attr)

    #     x = self.layers[0].conv(x, edge_index, edge_attr)

    #     for layer in self.layers[1:]:
    #         x = layer(x, edge_index, edge_attr)

    #     x = self.layers[0].act(self.layers[0].norm(x))
    #     x = F.dropout(x, p=0.1, training=self.training)
    #     return x