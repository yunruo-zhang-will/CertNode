# define GNN and smoothed GNN models
import math
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, APPNP, global_mean_pool
from torch_geometric.utils import to_dense_adj
from utils import count_arr, remove_nodes_pyg, remove_nodes_pyg_batch, remove_node_incident_edges_pyg_batch

## GNN models from AGNNCert

class NodeGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, hidden_size=20):
        super(NodeGCN, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index, edge_weight = None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
    
class SmoothNodeGCN(NodeGCN):
    def __init__(self, num_features, num_classes, hidden_size=20, config={'p_n': 0.5}, device=None):
        super(SmoothNodeGCN, self).__init__(num_features, num_classes, hidden_size)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)
        # print(f'self.p_n device: {self.p_n.device}')

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0
        adj_triu = torch.triu(adj_dense, diagonal=1)
        #a = torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n))
        #print(f'adj_triu device: {adj_triu.device}, p_n device: {self.p_n.device}, a device: {a.device} adj_dense device: {adj_dense.device}')
        adj_triu = adj_triu.to(self.device).mul(torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n.to(self.device))).to(self.device))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor, no_dense=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        # 
        # Similar to NodeAwareSmoothing, we use the original feature matrix
        # once a node is separated from other nodes, its feature can only affect its own classification
        # in our threat model, this means that the injected nodes are effectiveless 
        # because they cannot change the classfication of clean nodes anymore 
        with torch.no_grad():
            if no_dense:
                num_nodes = features.shape[0]
                edge_index, _ = remove_nodes_pyg(num_nodes, edge_index, self.p_n)
            else:
                adj_dense = torch.squeeze(to_dense_adj(edge_index))
                adj_dense = self.perturbation(adj_dense)
                edge_index = torch.nonzero(adj_dense).t()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num, no_dense=False):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features.to(self.device), edge_index.to(self.device), no_dense=no_dense).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]

        # test
        # print(f'top2.shape: {top2.shape}, count1.shape: {len(count1)}, count2.shape: {len(count2)}')
        # exit(0)
        # top2.shape: (3327, 2), count1.shape: 3327, count2.shape: 3327

        return top2, count1, count2

class GraphGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=32, dropout=0.5):
        super(GraphGCN, self).__init__()
        self.embedding_size = hidden_size * 3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.dropout3 = torch.nn.Dropout(dropout)
        self.lin = Linear(hidden_size * 3, num_classes)

    def forward(self, x, edge_index, batch):
        input_lin = self.embedding(x, edge_index, batch)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index, batch):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        out1 = self.dropout1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        out2 = self.dropout2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        out3 = self.dropout3(out3)
        stack.append(out3)

        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack, batch=batch)
        return input_lin

class SmoothGraphGCN(GraphGCN):
    def __init__(self, num_features, num_classes, hidden_size=20, dropout=0.5, config={'p_n': 0.5}, device=None):
        super(SmoothGraphGCN, self).__init__(num_features, num_classes, hidden_size, dropout)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def forward_perturb(self, features: Tensor, edge_index: Tensor, batch: Tensor, train=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        if train:
            p_n = 0.5 * self.p_n
        else:
            p_n = self.p_n
        with torch.no_grad():
            edge_index = remove_node_incident_edges_pyg_batch(
                edge_index,
                batch,
                p_n,
            )
        return self.forward(features, edge_index, batch)

    def smoothed_precit(self, features, edge_index, batch, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        num_graphs = int(batch.max().item()) + 1
        counts = np.zeros((num_graphs, self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index, batch).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        top2 = torch.as_tensor(top2, device=batch.device)
        return top2, count1, count2

class NodeGSAGE(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, hidden_size=20):
        super(NodeGSAGE, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index.to(torch.int64))
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class SmoothNodeGSAGE(NodeGSAGE):
    def __init__(self, num_features, num_classes, hidden_size=20, config={'p_n': 0.5}, device=None):
        super(SmoothNodeGSAGE, self).__init__(num_features, num_classes, hidden_size)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0
        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = adj_triu.to(self.device).mul(torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n.to(self.device))).to(self.device))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor, no_dense=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            if no_dense:
                num_nodes = features.shape[0]
                edge_index, _ = remove_nodes_pyg(num_nodes, edge_index, self.p_n)
            else:
                adj_dense = torch.squeeze(to_dense_adj(edge_index))
                adj_dense = self.perturbation(adj_dense)
                edge_index = torch.nonzero(adj_dense).t()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num, no_dense=False):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features.to(self.device), edge_index.to(self.device), no_dense=no_dense).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2

class GraphGSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=32):
        super(GraphGSAGE, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = SAGEConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = SAGEConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index, batch):
        input_lin = self.embedding(x, edge_index, batch)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index, batch):

        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack, batch=batch)

        return input_lin

class SmoothGraphGSAGE(GraphGSAGE):
    def __init__(self, num_features, num_classes, hidden_size=20, config={'p_n': 0.5}, device=None):
        super(SmoothGraphGSAGE, self).__init__(num_features, num_classes, hidden_size)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def forward_perturb(self, features: Tensor, edge_index: Tensor, batch: Tensor, train=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        if train:
            p_n = 0.5 * self.p_n
        else:
            p_n = self.p_n

        with torch.no_grad():
            edge_index = remove_node_incident_edges_pyg_batch(
                edge_index,
                batch,
                p_n,
            )
        return self.forward(features, edge_index, batch)

    def smoothed_precit(self, features, edge_index, batch, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        num_graphs = int(batch.max().item()) + 1
        counts = np.zeros((num_graphs, self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index, batch).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        top2 = torch.as_tensor(top2, device=batch.device)
        return top2, count1, count2

class NodeGAT(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes,hidden_size=20):
        super(NodeGAT, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(3*hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class SmoothNodeGAT(NodeGAT):
    def __init__(self, num_features, num_classes, hidden_size=20, config={'p_n': 0.5}, device=None):
        super(SmoothNodeGAT, self).__init__(num_features, num_classes, hidden_size)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0
        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = adj_triu.to(self.device).mul(torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n.to(self.device))).to(self.device))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor, no_dense=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            if no_dense:
                num_nodes = features.shape[0]
                edge_index, _ = remove_nodes_pyg(num_nodes, edge_index, self.p_n)
            else:
                adj_dense = torch.squeeze(to_dense_adj(edge_index))
                adj_dense = self.perturbation(adj_dense)
                edge_index = torch.nonzero(adj_dense).t()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num, no_dense=False):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features.to(self.device), edge_index.to(self.device), no_dense=no_dense).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2

class GraphGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes,hidden_size=32):
        super(GraphGAT, self).__init__()
        self.embedding_size=hidden_size*3
        self.conv1 = GATConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GATConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes)

    def forward(self, x, edge_index, batch):
        input_lin = self.embedding(x, edge_index, batch)
        final = self.lin(input_lin)
        return final
    
    def embedding(self, x, edge_index, batch):
            
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)
        
        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack, batch=batch)

        return input_lin

class SmoothGraphGAT(GraphGAT):
    def __init__(self, num_features, num_classes, hidden_size=20, config={'p_n': 0.5}, device=None):
        super(SmoothGraphGAT, self).__init__(num_features, num_classes, hidden_size)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def forward_perturb(self, features: Tensor, edge_index: Tensor, batch: Tensor, train=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        if train:
            p_n = 0.5 * self.p_n
        else:
            p_n = self.p_n

        with torch.no_grad():
            edge_index = remove_node_incident_edges_pyg_batch(
                edge_index,
                batch,
                p_n,
            )
        return self.forward(features, edge_index, batch)

    def smoothed_precit(self, features, edge_index, batch, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        num_graphs = int(batch.max().item()) + 1
        counts = np.zeros((num_graphs, self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index, batch).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        top2 = torch.as_tensor(top2, device=batch.device)
        return top2, count1, count2


#* additional GNN models: GIN, APPNP, which are not used in AGNNCert
# #TODO test these models

class NodeGIN(torch.nn.Module):
    """
    3 层 GIN 模型用于节点分类。
    """
    def __init__(self, num_features, num_classes, hidden_size=20):
        super(NodeGIN, self).__init__()
        self.embedding_size = hidden_size * 3
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        self.relu1 = ReLU()
        self.conv2 = GINConv(nn2)
        self.relu2 = ReLU()
        self.conv3 = GINConv(nn3)
        self.relu3 = ReLU()
        self.lin = Linear(3 * hidden_size, num_classes)

    def forward(self, x, edge_index):
        input_lin = self.embedding(x, edge_index)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)
        return input_lin


class SmoothNodeGIN(NodeGIN):
    def __init__(self, num_features, num_classes, hidden_size=20, config={'p_n': 0.5}, device=None):
        super(SmoothNodeGIN, self).__init__(num_features, num_classes, hidden_size)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0
        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = adj_triu.to(self.device).mul(torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n.to(self.device))).to(self.device))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor, no_dense=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            if no_dense:
                num_nodes = features.shape[0]
                edge_index, _ = remove_nodes_pyg(num_nodes, edge_index, self.p_n)
            else:
                adj_dense = torch.squeeze(to_dense_adj(edge_index))
                adj_dense = self.perturbation(adj_dense)
                edge_index = torch.nonzero(adj_dense).t()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num, no_dense=False):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features.to(self.device), edge_index.to(self.device), no_dense=no_dense).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2


class GraphGIN(torch.nn.Module):
    """
    3 层 GIN 模型用于图分类。
    """
    def __init__(self, num_features, num_classes, hidden_size=32, dropout=0.5):
        super(GraphGIN, self).__init__()
        self.embedding_size = hidden_size * 3
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        self.relu1 = ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv2 = GINConv(nn2)
        self.relu2 = ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GINConv(nn3)
        self.relu3 = ReLU()
        self.dropout3 = torch.nn.Dropout(dropout)
        self.lin = Linear(hidden_size * 3, num_classes)

    def forward(self, x, edge_index, batch):
        input_lin = self.embedding(x, edge_index, batch)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, batch):
        stack = []

        out1 = self.conv1(x, edge_index)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        out1 = self.dropout1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        out2 = self.dropout2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)
        out3 = self.dropout3(out3)
        stack.append(out3)

        stack = torch.cat(stack, dim=1)
        input_lin = global_mean_pool(stack, batch=batch)
        return input_lin


class SmoothGraphGIN(GraphGIN):
    def __init__(self, num_features, num_classes, hidden_size=20, dropout=0.5, config={'p_n': 0.5}, device=None):
        super(SmoothGraphGIN, self).__init__(num_features, num_classes, hidden_size, dropout)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def forward_perturb(self, features: Tensor, edge_index: Tensor, batch: Tensor, train=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        if train:
            p_n = 0.5 * self.p_n
        else:
            p_n = self.p_n

        with torch.no_grad():
            edge_index = remove_node_incident_edges_pyg_batch(
                edge_index,
                batch,
                p_n,
            )
        return self.forward(features, edge_index, batch)

    def smoothed_precit(self, features, edge_index, batch, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        num_graphs = int(batch.max().item()) + 1
        counts = np.zeros((num_graphs, self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index, batch).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        top2 = torch.as_tensor(top2, device=batch.device)
        return top2, count1, count2


class NodeAPPNP(torch.nn.Module):
    """
    APPNP 模型用于节点分类，包含 2 层 MLP 和 10 步 PPR。
    """
    def __init__(self, num_features, num_classes, hidden_size=64, K=10, alpha=0.1):
        super(NodeAPPNP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes)
        )
        self.appnp = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.appnp(x, edge_index)
        return x


class SmoothNodeAPPNP(NodeAPPNP):
    def __init__(self, num_features, num_classes, hidden_size=64, K=10, alpha=0.1, config={'p_n': 0.5}, device=None):
        super(SmoothNodeAPPNP, self).__init__(num_features, num_classes, hidden_size, K, alpha)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0
        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = adj_triu.to(self.device).mul(torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n.to(self.device))).to(self.device))
        adj_perted = adj_triu + adj_triu.t()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor, no_dense=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            if no_dense:
                num_nodes = features.shape[0]
                edge_index, _ = remove_nodes_pyg(num_nodes, edge_index, self.p_n)
            else:
                adj_dense = torch.squeeze(to_dense_adj(edge_index))
                adj_dense = self.perturbation(adj_dense)
                edge_index = torch.nonzero(adj_dense).t()
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num, no_dense=False):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features.to(self.device), edge_index.to(self.device), no_dense=no_dense).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2


class GraphAPPNP(torch.nn.Module):
    """
    APPNP 模型用于图分类，包含 2 层 MLP 和 10 步 PPR。
    """
    def __init__(self, num_features, num_classes, hidden_size=64, K=10, alpha=0.1):
        super(GraphAPPNP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes)
        )
        self.appnp = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index, batch):
        x = self.mlp(x)
        x = self.appnp(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


class SmoothGraphAPPNP(GraphAPPNP):
    def __init__(self, num_features, num_classes, hidden_size=64, K=10, alpha=0.1, config={'p_n': 0.5}, device=None):
        super(SmoothGraphAPPNP, self).__init__(num_features, num_classes, hidden_size, K, alpha)
        self.config = config
        self.device = device
        self.nclass = num_classes
        self.p_n = torch.tensor(config['p_n']).to(self.device)

    def forward_perturb(self, features: Tensor, edge_index: Tensor, batch, train=False) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        if train:
            p_n = 0.5 * self.p_n
        else:
            p_n = self.p_n

        with torch.no_grad():
            edge_index = remove_node_incident_edges_pyg_batch(
                edge_index,
                batch,
                p_n,
            )
        return self.forward(features, edge_index, batch)

    def smoothed_precit(self, features, edge_index, batch, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        num_graphs = int(batch.max().item()) + 1
        counts = np.zeros((num_graphs, self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index, batch).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2].copy()
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        top2 = torch.as_tensor(top2, device=batch.device)
        return top2, count1, count2


