# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:59:53 2024

@author: 31271
"""

import math
import torch
from torch_geometric.datasets import Planetoid, Amazon
import numpy as np
import scipy.sparse as sp
from numpy.random.mtrand import RandomState

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

def matri_to_index(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []
    
    for i in range(V):
        for j in range(i,V):
            if A[i,j]==1:
                edge_index_0.append(i)
                edge_index_1.append(j)
                if i!=j:
                    edge_index_0.append(j)
                    edge_index_1.append(i)
    return np.array([edge_index_0,edge_index_1])
                
def matri_to_index_directed(A):
    V = A.shape[0]
    edge_index_0 = []
    edge_index_1 = []
    
    for i in range(V):
        for j in range(V):
            if A[i,j]==1:
                edge_index_0.append(i)
                edge_index_1.append(j)
    return np.array([edge_index_0,edge_index_1])



#To load the data, we divide the train, validate and test dataset by amount for each class
#instead of ratio, to ensure a balanced training set  

def load_node_data(name, num_train=400, num_val=100, num_test=None, directed=False):
    if name == "CiteSeer" or name == "PubMed":
        dataset = Planetoid(root='./' + name + '/', name=name,num_train_per_class=50)
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        num_classes = dataset.num_classes

    elif name == "computers":
        dataset = Amazon(root='./' + name + '/', name=name)
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _device = torch.device('cpu')
        data = dataset[0].to(_device)
        data.train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        data.test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
        
        num_classes = dataset.num_classes
    elif name == "Cora-ML":
        # 指定数据文件路径
        path = './datasets/cora_ml.npz'
        graph = np.load(path)
        # sp.csr_matrix 用于高效存储和操作稀疏矩阵，常见于图神经网络的数据预处理阶段。
        A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
        adj = A.toarray()
        edge_index = torch.nonzero(torch.tensor(adj)).t()
        data = (np.ones(graph['X'].shape[1]), graph['X'])
        X = sp.csr_matrix(data, dtype=np.float32)
        x = X.toarray()
        y = graph['y']
        '''data_name = './datasets/cora_ml.npz'
        # 加载 npz 文件，包含稀疏矩阵的各项数据
        with np.load(data_name, allow_pickle = True) as loader:
            loader = dict(loader)
            # 构造邻接矩阵 A（稀疏格式），再转为普通 numpy 数组
            A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                               loader['adj_indptr']), shape=loader['adj_shape'])
            adj = A.toarray()
            # 构造属性矩阵 X（稀疏格式），再转为普通 numpy 数组
            X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
            x = X.toarray()
            # 读取节点标签
            y = loader.get('labels')
            # 根据 directed 参数选择边索引文件
            if not directed:
                edge_index = np.load("core_ml_edge_index.npy")  # 无向图的边索引
            else:
                edge_index = np.load("core_ml_edge_index_d.npy")  # 有向图的边索引'''
        # 构造 torch_geometric 的 Data 对象，包含节点特征、边索引和标签
        data = Data(x=torch.tensor(x,dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y))
        # 初始化训练、验证、测试掩码（全为 False）
        data.train_mask = torch.zeros(y.size, dtype=torch.bool)
        data.val_mask = torch.zeros(y.size, dtype=torch.bool)
        data.test_mask = torch.zeros(y.size, dtype=torch.bool)
        # 统计类别数
        num_classes = len(np.unique(y))

    
  
    prng = RandomState(12) # Make sure that the permutation is always the same, even if we set the seed different
    
    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = prng.permutation(idx)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train+num_val]
        if num_test is None:
            test_idx = idx[num_train+num_val:-1]
        
        else:
            test_idx = idx[num_train+num_val:num_train+num_val+num_test]
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    else:
        num_node_features = data.x.shape[1]
    return data, num_node_features, num_classes


def load_graph_data(name, num_train=100, num_val=50, num_test=50, directed=False):
    
    graphs = TUDataset(root="./datasets/", name=name, use_node_attr=True)
    num_node_features = graphs[0].x.shape[1]
    #print(graphs[0].x.shape)
    #print(graphs[0].y.shape)
    #print(graphs[0].edge_index.shape)
    ys = [graphs[i].y.item() for i in range(len(graphs))]
    num_classes = len(np.unique(ys))
    seed=12
    rng = np.random.RandomState(seed)
    
    train_mask = torch.zeros(len(graphs), dtype=torch.bool)
    val_mask = torch.zeros(len(graphs), dtype=torch.bool)
    test_mask = torch.zeros(len(graphs), dtype=torch.bool)
    
    for c in range(num_classes):
        idx = (torch.tensor(ys) == c).nonzero(as_tuple=False).view(-1)
        idx = rng.permutation(idx)
        num_all = len(idx)
        num_train = math.floor(num_all*0.6)
        num_val = math.floor(num_all*0.2)
        train_idx = idx[:num_train]
        val_idx = idx[num_train:num_train+num_val]
        test_idx = idx[num_train+num_val:-1]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    # graphs: TUDataset, num_node_features: int, num_classes: int, mask_split: list of three boolean tensors, ys: list of int
    return graphs, num_node_features, num_classes, [train_mask, val_mask, test_mask], ys

