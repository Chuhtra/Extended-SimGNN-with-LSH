import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers import GatedGCN

class GatedGCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = 70
        n_classes = net_params['n_classes']
        dropout = 0.0
        n_layers = 4
        self.graph_norm = True
        self.batch_norm = True
        self.residual = True
        self.n_classes = n_classes

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GatedGCN(hidden_dim, hidden_dim, dropout,
                                                   self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers)])
        # trainer.MLP_layer = MLPReadout(hidden_dim, n_classes, L=0)

    def forward(self, g, h, e, snorm_n, snorm_e):
        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e, snorm_n.unsqueeze(-1), snorm_e.unsqueeze(-1))

        # output
        h_out = self.MLP_layer(h)

        return h_out
