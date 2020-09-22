import torch
import networkx
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
import utils
from utils import calculate_ranking_correlation, calculate_prec_at_k, denormalize_sim_score

from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_node_labels, number_of_edge_labels):
        """
        :param args: Arguments object.
        :param number_of_node_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_node_labels = number_of_node_labels
        # self.number_edge_labels = number_of_edge_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_node_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_node_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        # elif self.args.gnn_operator == 'gatedgcn':
        #    self.convolution_1 = GatedGCN(self.number_node_labels, self.args.filters_1)
        #    self.convolution_2 = GatedGCN(self.args.filters_1, self.args.filters_2)
        #    self.convolution_3 = GatedGCN(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gat':
            self.convolution_1 = GATConv(self.number_node_labels, self.args.filters_1)
            self.convolution_2 = GATConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GATConv(self.args.filters_2, self.args.filters_3)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        if self.args.diffpool:
            self.attention = DiffPool(self.args)
        else:
            self.attention = AttentionModule(self.args)

        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2, batch_1, batch_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histsogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(abstract_features_1, abstract_features_2.permute([0, 2, 1])).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[:num_nodes[i], :num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args.bins)
            hist = hist / torch.sum(hist)
            hist = hist.view(1, -1)
            hist_list.append(hist)

        return torch.stack(hist_list).view(-1, self.args.bins)

    def convolutional_pass(self, edge_index, features, edge_attr=None):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        '''
        if self.args.gnn_operator == 'gatedgcn':
            edge_features, node_features = self.convolution_1(features, edge_index, edge_attr)
            node_features = F.relu(node_features)
            node_features = F.dropout(node_features, p=self.args.dropout, training=self.training)
            edge_features, node_features = self.convolution_2(node_features, edge_index, edge_attr)
            node_features = F.relu(node_features)
            node_features = F.dropout(node_features, p=self.args.dropout, training=self.training)
            _, features = self.convolution_3(node_features, edge_index, edge_attr)
        else:
        Caution: If the 'if' statement is utilized fix identation.
        '''

        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index)

        return features

    def diffpool(self, abstract_features, edge_index, batch):
        """
        Making differentiable pooling.
        :param abstract_features: Node feature matrix.
        :param batch: Batch vector, which assigns each node to a specific example
        :return pooled_features: Graph feature matrix.
        """
        x, mask = to_dense_batch(abstract_features, batch)
        adj = to_dense_adj(edge_index, batch)
        return self.attention(x, adj, mask)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """

        # Graph attribute initiallizations
        g1 = data["g1"]
        g2 = data["g2"]

        pooled_features_1, abstract_features_1, batch_1 = self.get_embedding(g1)
        pooled_features_2, abstract_features_2, batch_2 = self.get_embedding(g2)

        # Output scores' vector
        scores = self.tensor_network(pooled_features_1, pooled_features_2)

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1, abstract_features_2, batch_1, batch_2)
            scores = torch.cat((scores, hist), dim=1)

        # Final connected layers
        scores = F.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)

        return score

    def get_embedding(self, g):
        """
        This method implements the first 2 steps of the pipeline for a given graph. It used to be directly implemented
        in the forward() method, but due to the needs of LSH utilization I detached it. It returns intermediate data too,
        because they are needed from "Strategy 2" module.
        :param g: The graph we want to take the embedding for.
        :return: The embedding of the given graph g, abstract_features(node embeddings) and batch.
        """
        # Graph attribute initiallizations
        edge_index = g.edge_index

        # edge_attr = g.edge_attr
        features = g.x
        batch = g.batch if hasattr(g, 'batch') else torch.tensor((), dtype=torch.long).new_zeros(g.num_nodes)

        # Node embeddings
        '''
        if self.args.gnn_operator == 'gatedgcn':
            if self.number_edge_labels == 0:
                edge_attr_1 = torch.ones(edge_attr_1.size(0), 1)
                edge_attr_2 = torch.ones(edge_attr_2.size(0), 1)

            abstract_features_1 = self.convolutional_pass(edge_index_1, features_1, edge_attr_1)
            abstract_features_2 = self.convolutional_pass(edge_index_2, features_2, edge_attr_2)
        else:
        Caution: If the 'if' statement is utilized fix indentation.
        '''
        abstract_features = self.convolutional_pass(edge_index, features)

        # Graph embeddings
        if self.args.diffpool:
            graph_embedding = self.diffpool(abstract_features, edge_index, batch)
        else:
            graph_embedding = self.attention(abstract_features, batch)

        return graph_embedding, abstract_features, batch
