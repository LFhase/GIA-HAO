"""
A general graph dataset loading component
"""
from operator import mod
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import os
import numpy as np
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import platform
import grb
from grb.dataset import Dataset

if "windows" in platform.system().lower():
    base_dir = "E:/.datasets"
else:
    base_dir = "../.datasets"


def load_graph_dataset(name, mode="full", self_loop=True, undirected=True):
    if name == "reddit":
        return load_reddit()
    elif name == "cora":
        return load_cora()
    elif name == 'dblp':
        return load_dblp(pyg=False,self_loop=self_loop)
    elif name.lower() in ["texas","wisconsin","cornell","actor","squirrel","chameleon"]:
        return load_heter_g(name)
    elif name.lower().startswith("grb-"):
        return load_grb(name, mode, self_loop)
    else:
        return load_ogb(name, self_loop, undirected)

def load_grb(name, mode="full", self_loop=True):
    dataset = Dataset(name,
                    data_dir=os.path.join(base_dir,"grb",name),
                    mode=mode,feat_norm="arctan")
    edge_index = torch.LongTensor(dataset.adj.nonzero())
    num_classes = dataset.num_classes
    
    # convert to dgl format
    graph = dgl.graph((edge_index[0], edge_index[1])).to_simple()
    graph = dgl.to_bidirected(graph)

    if self_loop:
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

    graph.ndata['features'] = dataset.features
    graph.ndata['labels'] = dataset.labels
    graph.ndata['train_mask'] = dataset.train_mask
    graph.ndata['val_mask'] = dataset.val_mask
    graph.ndata['test_mask'] = dataset.test_mask

    return graph, num_classes

def load_cora(self_loop=True):
    from dgl.data import CoraGraphDataset
    data = CoraGraphDataset(raw_dir=base_dir)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    if self_loop:
        g = g.remove_self_loop()
        g = g.add_self_loop()
    return g, data.num_classes

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(raw_dir=base_dir, self_loop=self_loop)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes


def load_ogb(name, self_loop=True, undirected=True):

    print('load', name)
    data = DglNodePropPredDataset(name=name, root=base_dir)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]
    
    if self_loop:
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        if undirected:
            graph = dgl.to_bidirected(graph.to_simple(), copy_ndata=True)
        # if "arxiv" not in name:
        #     graph = graph.to_simple()
    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    # type fix for arxiv_2, see https://github.com/dmlc/dgl/pull/1987/commits/b45a5c8f2916c1ca61945b8b1efe9d03893f2a65
    from dgl.data.utils import generate_mask_tensor
    g.ndata['train_mask'] = generate_mask_tensor(g.ndata['train_mask'].numpy())
    g.ndata['val_mask'] = generate_mask_tensor(g.ndata['val_mask'].numpy())
    g.ndata['test_mask'] = generate_mask_tensor(g.ndata['test_mask'].numpy())
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g


import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T


def load_dblp(pyg=True, self_loop=True):
    dataset = CitationFull(base_dir, 'dblp', transform=T.NormalizeFeatures())
    data = dataset[0]
    num_classes = dataset.num_classes

    train_mask, val_mask, test_mask = generate_percent_split(dataset, seed=0, train_percent=10, val_percent=10)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    if not pyg:
        graph = dgl.graph((data.edge_index[0], data.edge_index[1])).to_simple()
        graph = dgl.to_bidirected(graph)

        if self_loop:
            graph = graph.remove_self_loop()
            graph = graph.add_self_loop()

        graph.ndata['features'] = torch.Tensor(data.x)
        graph.ndata['labels'] = torch.LongTensor(data.y)
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
    else:
        graph = data
    return graph, num_classes

# from https://github.com/yang-han/P-reg/blob/master/utils.py

class Mask(object):
    def __init__(self, train_mask, val_mask, test_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

def generate_percent_split(dataset, seed=0, train_percent=10, val_percent=10):
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = torch.nonzero(data.y == c,as_tuple=True)[0].flatten()
        num_c = all_c_idx.size(0)
        train_num_per_c = num_c * train_percent // 100
        val_num_per_c = num_c * val_percent // 100
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask

def generate_split(dataset, seed=0, train_num_per_c=20, val_num_per_c=30):
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask

def generate_grb_split(data,mode='full'):
    # data = dataset[0]
    # num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    adj = data.adj_t
    degs = adj.sum(-1)
    _, idx = degs.sort()
    n_total = len(idx)
    n_out = int(n_total*0.05)
    n_cag = int((n_total-n_out)*0.3)
    easy_idx = idx[n_out:n_out+n_cag]
    med_idx = idx[n_out+n_cag:n_out+n_cag+n_cag]
    hard_idx = idx[n_out+n_cag+n_cag:n_out+n_cag+n_cag+n_cag]
    esel_idx = torch.randperm(n_cag)
    msel_idx = torch.randperm(n_cag)
    hsel_idx = torch.randperm(n_cag)
    n_test = int(n_total*0.1)

    if mode.lower() == "full":
        test_mask[easy_idx[esel_idx[:n_test]]] = 1
        test_mask[med_idx[msel_idx[:n_test]]] = 1
        test_mask[hard_idx[hsel_idx[:n_test]]] = 1
    elif mode.lower() == "easy":
        test_mask[easy_idx[esel_idx[:n_test]]] = 1
    elif mode.lower() == "medium":
        test_mask[med_idx[msel_idx[:n_test]]] = 1
    elif mode.lower() == "hard":
        test_mask[hard_idx[hsel_idx[:n_test]]] = 1
    else:
        raise Exception("no such mode")
    left_idx = torch.nonzero(torch.logical_not(torch.logical_or(test_mask,train_mask)),as_tuple=True)[0]
    random_idx = torch.randperm(len(left_idx))
    n_train = int(len(idx)*0.6)
    train_mask[left_idx[random_idx[:n_train]]] = 1
    val_mask[left_idx[random_idx[n_train:]]] = 1
    print(f"split datasets into train {train_mask.sum()}/{n_total}, deg {degs[test_mask].mean()}")
    print(f"                      val {val_mask.sum()}/{n_total}, deg {degs[test_mask].mean()}")
    print(f"                     test {test_mask.sum()}/{n_total}, deg {degs[test_mask].mean()}")
    return train_mask, val_mask, test_mask

def load_split(path):
    mask = torch.load(path)
    return mask.train_mask, mask.val_mask, mask.test_mask

import os
import re

import networkx as nx
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit

import utils

def load_heter_g(dataset_name, pyg=False, splits_file_path=None, train_percentage=0.48, val_percentage=0.32, embedding_mode=None,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = utils.load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('data', dataset_name,
                                                                f'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = utils.preprocess_features(features)

    if not embedding_mode:
        g = dgl.from_scipy(adj + sp.eye(adj.shape[0]))
    else:
        if embedding_mode == 'ExperimentTwoAll':
            embedding_file_path = os.path.join('embedding_method_combinations_all',
                                               f'outf_nodes_relation_{dataset_name}all_embedding_methods.txt')
        elif embedding_mode == 'ExperimentTwoPairs':
            embedding_file_path = os.path.join('embedding_method_combinations_in_pairs',
                                               f'outf_nodes_relation_{dataset_name}_graph_{embedding_method_graph}_space_{embedding_method_space}.txt')
        else:
            embedding_file_path = os.path.join('structural_neighborhood',
                                           f'outf_nodes_space_relation_{dataset_name}_{embedding_method}.txt')
        space_and_relation_type_to_idx_dict = {}

        with open(embedding_file_path) as embedding_file:
            for line in embedding_file:
                if line.rstrip() == 'node1,node2	space	relation_type':
                    continue
                line = re.split(r'[\t,]', line.rstrip())
                assert (len(line) == 4)
                assert (int(line[0]) in G and int(line[1]) in G)
                if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                    space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                        space_and_relation_type_to_idx_dict)
                if G.has_edge(int(line[0]), int(line[1])):
                    G.remove_edge(int(line[0]), int(line[1]))
                G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                    (line[2], int(line[3]))])

        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)
        for node in sorted(G.nodes()):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        g = DGLGraph(adj)

        for u, v, feature in G.edges(data='subgraph_idx'):
            g.edges[g.edge_id(u, v)].data['subgraph_idx'] = th.tensor([feature])

    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        assert (train_percentage is not None and val_percentage is not None)
        assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

        if dataset_name in {'cora', 'citeseer'}:
            disconnected_node_file_path = os.path.join('unconnected_nodes', f'{dataset_name}_unconnected_nodes.txt')
            with open(disconnected_node_file_path) as disconnected_node_file:
                disconnected_node_file.readline()
                disconnected_nodes = []
                for line in disconnected_node_file:
                    line = line.rstrip()
                    disconnected_nodes.append(int(line))

            disconnected_nodes = np.array(disconnected_nodes)
            connected_nodes = np.setdiff1d(np.arange(features.shape[0]), disconnected_nodes)

            connected_labels = labels[connected_nodes]

            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(connected_labels), connected_labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(connected_labels[train_and_val_index]), connected_labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[connected_nodes[train_index]] = 1
            val_mask = np.zeros_like(labels)
            val_mask[connected_nodes[val_index]] = 1
            test_mask = np.zeros_like(labels)
            test_mask[connected_nodes[test_index]] = 1
        else:
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(labels), labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[train_index] = 1
            val_mask = np.zeros_like(labels)
            val_mask[val_index] = 1
            test_mask = np.zeros_like(labels)
            test_mask[test_index] = 1

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    # not needed here
    # degs = g.in_degrees().float()
    # norm = th.pow(degs, -0.5)#.cuda()
    # norm[th.isinf(norm)] = 0
    # g.ndata['norm'] = norm.unsqueeze(1)
    # if self_loop:
    
    g = g.remove_self_loop()
    g = g.add_self_loop()
        # if undirected:
    g = dgl.to_bidirected(g.to_simple(), copy_ndata=True)

    g.ndata['features'] = features
    g.ndata['labels'] = labels

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    if pyg:
        # print(torch.stack([g.edges()[0],g.edges()[1]],dim=0).size())
        data = Data(x=features,edge_index=torch.stack([g.edges()[0],g.edges()[1]],dim=0),
                    y=labels)
        # data.num_nodes = features.size(0)
        # data.num_edges = g.edges()[0].size(0)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        g = data

    return g, num_labels


def load_grb(name, mode="full"):
    dataset = Dataset(name,
                    data_dir=os.path.join(base_dir,"grb",name),
                    mode=mode,feat_norm="") #do feature normalization later
    data = Data(edge_index=torch.LongTensor(dataset.adj.nonzero()),
                x=dataset.features,y=dataset.labels)
    
    data.train_mask = dataset.train_mask
    data.val_mask = dataset.val_mask
    data.test_mask = dataset.test_mask
    data = T.ToSparseTensor()(data)
    print(data.adj_t.is_symmetric())
    data.adj_t = data.adj_t.to_symmetric()
    num_classes = dataset.num_classes
    return data, num_classes
