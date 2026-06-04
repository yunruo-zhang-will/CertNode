import numpy as np
import torch
import torch.nn as nn
import os
import random
import scipy.sparse as sp
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, subgraph
from typing import List

def init_random_seed(SEED=2021):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
init_random_seed()

def load_data(path):
    graph = np.load(path)
    A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
    data = (np.ones(graph['X'].shape[1]), graph['X'])
    X = sp.csr_matrix(data, dtype=np.float32).todense()
    y = graph['y']
    n, d = X.shape
    nc = y.max() + 1
    return A, X, y, n, d, nc

def get_degrees(edge_index):
    adj_dense = torch.squeeze(to_dense_adj(edge_index))
    adj_dense.fill_diagonal_(0)
    (adj_dense==adj_dense.T).all()
    degrees = adj_dense.sum(0).cpu().numpy().astype(np.int16)
    return degrees

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split(labels, n_per_class=20, seed=0):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [n_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [n_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    np.random.seed(seed)
    nc = labels.max() + 1
    nc = int(nc)
    # n_per_class: 50 (id: 140465346922256, type: <class 'int'>)
    # nc: 6 (id: 140460569073392, type: <class 'numpy.int8'>)
    # numpy.int8 与 Python 内置 int 类型相乘时发生了溢出问题。

    """# 立即检查乘法结果
    expected_size = n_per_class * nc
    print(f"DEBUG: Expected size = {n_per_class} * {nc} = {expected_size}")
    
    print("=== 深入调试 ===")
    print(f"n_per_class: {n_per_class} (id: {id(n_per_class)}, type: {type(n_per_class)})")
    print(f"nc: {nc} (id: {id(nc)}, type: {type(nc)})")
    
    import operator
    # 多种方式计算
    result1 = n_per_class * nc
    result2 = int.__mul__(n_per_class, nc)
    result3 = operator.mul(n_per_class, nc)
    
    print(f"直接乘法: {result1}")
    print(f"int.__mul__: {result2}")
    print(f"operator.mul: {result3}")
    
    # 检查是否被猴子补丁
    print(f"int.__mul__ is {int.__mul__}")
    print(f"原始int.__mul__: {int.__mul__ is int.__mul__}")"""

    split_train, split_val = [], []
    for l in range(nc):
        np.random.seed(seed+l)
        perm = np.random.RandomState(seed=seed).permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])


    split_train = np.random.RandomState(seed=seed).permutation(np.concatenate(split_train))
    split_val = np.random.RandomState(seed=seed).permutation(np.concatenate(split_val))

    #print(f'split_train.shape[0]: {split_train.shape[0]}, split_val.shape[0]: {split_val.shape[0]}, n_per_class * nc: {n_per_class * nc}, n_per_class: {n_per_class}, nc: {nc}')
    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc
    """assert split_train.shape[0] == split_val.shape[0]
    result = n_per_class * nc
    print(f'n_per_class * nc: {result}, n_per_class: {n_per_class}, nc: {nc}')
    assert split_val.shape[0] == n_per_class * nc
    # split_train.shape[0]: 300, split_val.shape[0]: 300, n_per_class: 50, nc: 6"""

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))
    print("Number of samples per class:", n_per_class)
    print("Training-validation-testing Size:", len(split_train),len(split_val),len(split_test))
    return split_train, split_val, split_test

def normalize(adj):
    degree = torch.sum(adj,dim=0)
    D_half_norm = torch.pow(degree, -0.5)
    D_half_norm = torch.nan_to_num(D_half_norm, nan=0.0, posinf=0.0, neginf=0.0)
    D_half_norm = torch.diag(D_half_norm)
    DAD = torch.mm(torch.mm(D_half_norm,adj), D_half_norm)
    return DAD

def count_arr(predictions, nclass):
    nodes_n=predictions.shape[0]
    counts = np.zeros((nodes_n,nclass), dtype=int)
    for n,idx in enumerate(predictions):
        counts[n,idx] += 1
    return counts

def listSubset(A_list,index_list):
    '''take out the elements of a list (A_list) by a index list (index_list)'''
    return [A_list[i] for i in index_list]

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def data_to_adj_feat_labels(data, directed=None):
    """
    将 torch_geometric.data.Data 对象转换为邻接矩阵、特征矩阵和标签。
    返回: adjacency_matrix (torch.Tensor), feature_matrix (torch.Tensor), labels (torch.Tensor)

    torch_geometric.data.Data 对象，包含如下常用属性：
    data.x：节点特征矩阵，形状为 [num_nodes, num_features]，类型为 torch.Tensor。
    data.edge_index：图的边列表，形状为 [2, num_edges]，类型为 torch.LongTensor，每一列表示一条边的起止节点。
    data.y：节点标签，形状为 [num_nodes]，类型为 torch.LongTensor。
    data.train_mask、data.val_mask、data.test_mask：布尔类型的掩码，用于区分训练、验证和测试集上的节点
    """
    # 特征矩阵
    feature_matrix = data.x
    # 标签
    labels = data.y
    # 邻接矩阵（稠密）
    adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float)
    adj[data.edge_index[0], data.edge_index[1]] = 1.0

    if directed is None:
        directed = False
        edges = set((int(i), int(j)) for i, j in data.edge_index.t().tolist())
        for i, j in edges:
            if (j, i) not in edges:
                directed = True

    # 如果是无向图，补全对称
    if directed == False:
        adj = adj + adj.t()
        adj[adj > 1] = 1
    return adj, feature_matrix, labels

def get_mask_indices(test_mask):
    """
    将 test_mask 转换为一维 numpy 数组，包含测试集节点的索引。
    支持 test_mask 为布尔数组或索引数组。
    """

    test_mask = np.array(test_mask)
    if test_mask.dtype == bool:
        return np.where(test_mask)[0]
    else:
        return test_mask.flatten()

def remove_nodes_pyg(num_nodes, edge_index, p, relabel_nodes=False, return_mask=False, min_nodes=2):
    """
    使用PyTorch Geometric的版本
    以概率p删除图中的节点，并移除与之相关的所有边
    
    参数:
    num_nodes: 图中节点总数
    edge_index: 边的索引，shape为[2, num_edges]
    p: 删除节点的概率 (0 <= p <= 1)
    return_mask: 是否返回节点保留掩码
    
    返回:
    新的edge_index，可能还有节点保留掩码
    """
    device = edge_index.device
    # 生成保留节点的掩码
    #print(f'num_nodes: {num_nodes}')
    #print(f'edge_index.device: {edge_index.device}')
    #print(f'torch.rand(num_nodes, device=device): {torch.rand(num_nodes, device=device)}')
    keep_mask = torch.rand(num_nodes, device=device) > p
    remaining_nodes = torch.where(keep_mask)[0]

    # keep_mask = torch.rand(num_nodes, device=device) > 2

    # 如果keep_mask中没有True值，强制保留至少min_nodes个节点
    if keep_mask.sum() < min_nodes:
        #print(f'keep_mask.sum() < min_nodes: {keep_mask.sum()} < {min_nodes}')
        num_to_add = min_nodes - keep_mask.sum().item()
        all_indices = torch.arange(num_nodes, device=device)
        false_indices = all_indices[~keep_mask]
        if len(false_indices) > 0:
            indices_to_add = false_indices[torch.randperm(len(false_indices))[:num_to_add]]
            keep_mask[indices_to_add] = True
            remaining_nodes = torch.where(keep_mask)[0]
    
    # print(f'Number of nodes after removal: {keep_mask.sum().item()} (min_nodes={min_nodes})')
    # exit(0)
    
    # 使用subgraph函数提取子图
    new_edge_index, _ = subgraph(
        remaining_nodes, 
        edge_index, 
        num_nodes=num_nodes, 
        relabel_nodes=relabel_nodes
    )

    if return_mask:
        return new_edge_index, keep_mask
    
    return new_edge_index, _


def remove_nodes_pyg_batch(edge_index, batch, p, relabel_nodes=False, return_mask=False, min_nodes_per_graph=1):
    """
    Batch-aware node removal for graph classification.
    Ensures every graph in the batch keeps at least ``min_nodes_per_graph`` nodes,
    so graph-level pooling still produces one embedding per original graph.
    """
    device = batch.device
    num_nodes = batch.size(0)
    keep_mask = torch.rand(num_nodes, device=device) > p

    if batch.numel() > 0:
        num_graphs = int(batch.max().item()) + 1
        for graph_idx in range(num_graphs):
            graph_nodes = torch.where(batch == graph_idx)[0]
            if graph_nodes.numel() == 0:
                continue
            kept_in_graph = int(keep_mask[graph_nodes].sum().item())
            if kept_in_graph >= min_nodes_per_graph:
                continue
            missing = min_nodes_per_graph - kept_in_graph
            dropped_graph_nodes = graph_nodes[~keep_mask[graph_nodes]]
            if dropped_graph_nodes.numel() == 0:
                continue
            add_idx = dropped_graph_nodes[torch.randperm(dropped_graph_nodes.numel(), device=device)[:missing]]
            keep_mask[add_idx] = True

    remaining_nodes = torch.where(keep_mask)[0]
    new_edge_index, _ = subgraph(
        remaining_nodes,
        edge_index,
        num_nodes=num_nodes,
        relabel_nodes=relabel_nodes,
    )

    if return_mask:
        return new_edge_index, keep_mask

    return new_edge_index, _


def remove_node_incident_edges_pyg_batch(edge_index, batch, p, return_mask=False, min_kept_nodes_per_graph=1):
    """
    Sample nodes independently in each graph batch and remove every edge incident to
    sampled nodes, while keeping all node features and batch assignments unchanged.
    This weakens message passing without dropping graph-level outputs.
    """
    device = batch.device
    num_nodes = batch.size(0)
    keep_node_mask = torch.rand(num_nodes, device=device) > p

    if batch.numel() > 0:
        num_graphs = int(batch.max().item()) + 1
        for graph_idx in range(num_graphs):
            graph_nodes = torch.where(batch == graph_idx)[0]
            if graph_nodes.numel() == 0:
                continue
            kept_in_graph = int(keep_node_mask[graph_nodes].sum().item())
            if kept_in_graph >= min_kept_nodes_per_graph:
                continue
            missing = min_kept_nodes_per_graph - kept_in_graph
            dropped_graph_nodes = graph_nodes[~keep_node_mask[graph_nodes]]
            if dropped_graph_nodes.numel() == 0:
                continue
            add_idx = dropped_graph_nodes[torch.randperm(dropped_graph_nodes.numel(), device=device)[:missing]]
            keep_node_mask[add_idx] = True

    edge_keep_mask = keep_node_mask[edge_index[0]] & keep_node_mask[edge_index[1]]
    new_edge_index = edge_index[:, edge_keep_mask]

    if return_mask:
        return new_edge_index, keep_node_mask

    return new_edge_index

# 批量处理版本（适用于多个图）
def batch_remove_nodes(batch, edge_index, p):
    """
    处理批量数据的版本
    以概率p删除图中的节点，并移除与之相关的所有边
    
    参数:
    num_nodes: 图中节点总数
    edge_index: 边的索引，shape为[2, num_edges]
    p: 删除节点的概率 (0 <= p <= 1)
    return_mask: 是否返回节点保留掩码
    
    返回:
    新的edge_index，可能还有节点保留掩码
    """
    if hasattr(batch, 'batch_size'):
        num_nodes = batch.num_nodes
    else:
        num_nodes = batch.max().item() + 1
    
    return remove_nodes_pyg(num_nodes, edge_index, p)

def split_train_mask(
        train_mask: List[torch.Tensor], 
        split_ratio: float = 0.05, 
        num_splits: int = 5
    ) -> List[List[torch.Tensor]]:
    """
    从train_mask中挑选出指定比例的元素，然后平均分为若干份
    
    Args:
        train_mask: 输入列表，包含torch.Size([])形状的Tensor
        split_ratio: 挑选比例，默认为5% (0.05)
        num_splits: 要分成的份数，默认为5
    
    Returns:
        包含num_splits个子列表的列表，每个子列表包含大致相等数量的Tensor
    """
    # 验证输入
    if not train_mask:
        return [[] for _ in range(num_splits)]
    
    if split_ratio <= 0 or split_ratio > 1:
        raise ValueError("split_ratio必须在0到1之间")
    
    if num_splits <= 0:
        raise ValueError("num_splits必须大于0")
    
    # 计算要挑选的元素数量
    total_elements = len(train_mask)
    num_to_select = max(1, int(total_elements * split_ratio))
    
    # 如果挑选数量少于份数，调整份数
    if num_to_select < num_splits:
        num_splits = num_to_select
        print(f"警告: 挑选数量({num_to_select})少于要求份数，调整为{num_splits}份")
    
    # 随机挑选指定比例的元素
    selected_indices = random.sample(range(total_elements), num_to_select)
    selected_elements = [train_mask[i] for i in selected_indices]
    
    # 计算每份应该包含的元素数量
    elements_per_split = num_to_select // num_splits
    remainder = num_to_select % num_splits
    
    # 将选中的元素平均分成num_splits份
    result = []
    start_index = 0
    
    for i in range(num_splits):
        # 计算当前份的元素数量（处理余数分配）
        current_size = elements_per_split + (1 if i < remainder else 0)
        
        # 提取当前份的元素
        end_index = start_index + current_size
        current_split = selected_elements[start_index:end_index]
        
        result.append(current_split)
        start_index = end_index
    
    return result