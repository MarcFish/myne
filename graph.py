import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from utils import read_csv, Vocab, read_txt
import collections
from pathlib import Path
import typing


class StaticGraph:
    def __init__(self):
        self._adj = None
        self._adj_csr = None
        self._vocab = None
        self._label_vocab = None
        self._node_label = None
        self._node_feature = None

    @property
    def adj(self):
        return self._adj

    @property
    def adj_csr(self):
        return self._adj_csr

    @adj.setter
    def adj(self, x):
        self._adj = x
        self._adj_csr = self._adj.tocsr()

    @property
    def node_size(self):
        return self._adj.shape[0]

    @property
    def edge_size(self):
        return self._adj.nnz // 2

    @property
    def vocab(self):
        return self._vocab

    @property
    def node_array(self):
        return np.arange(0, self._adj.shape[0])

    @property
    def edge_array(self):
        return np.stack((self._adj.row, self._adj.col), axis=-1)

    def read_node_feature(self, feature_dict: dict):
        feature_list = [None] * self.node_size
        for n, feature in feature_dict.items():
            n = self.vocab.stoi[n]
            feature_list[n] = feature
        feature_array = np.asarray(feature_list, dtype=np.float32)
        self._node_feature = feature_array

    def read_node_label(self, label_dict: dict):
        label_list = [None] * self.node_size
        labels = []
        for label in label_dict.values():
            labels.append(label)
        self._label_vocab = Vocab(collections.Counter(labels))
        for n, label in label_dict.items():
            n = self.vocab.stoi[n]
            label = self._label_vocab.stoi[label]
            label_list[n] = label
        self._node_label = np.asarray(label_list, dtype=np.int32)

    def read_edge(self, filename: Path):
        if "txt" in filename.suffix:
            read_func = read_txt
        elif "csv" in filename.suffix:
            read_func = read_csv
        else:
            read_func = read_txt
        node_list = list()
        for row in read_func(filename):
            node_list.append(row[0])
            node_list.append(row[1])
        self._vocab = Vocab(collections.Counter(node_list))
        edge_array = []
        for row in read_func(filename):
            n1 = self._vocab.stoi[row[0]]
            n2 = self._vocab.stoi[row[1]]
            edge_array.append([n1, n2])
            edge_array.append([n2, n1])
        edge_array = np.asarray(edge_array, dtype=np.int32)
        self._adj = coo_matrix((np.ones(len(edge_array)), (edge_array[:, 0], edge_array[:, 1])), shape=(len(self._vocab), len(self._vocab)))
        self._adj_csr = self._adj.tocsr() + sp.eye(len(self._vocab))
        self._adj = self._adj_csr.tocoo()

    def get_node_neighbors(self, node):
        return self._adj_csr[node].indices

    def get_node_degree(self, node):
        return self._adj_csr[node].nnz

    def get_nodes_degree_list(self):
        return np.asarray([self.get_node_degree(n) for n in self.node_array])

    def get_nodes_label(self, nodes: typing.Union[list, np.ndarray, None]) -> np.ndarray:
        if nodes is None:
            return self._node_label
        else:
            return self._node_label[nodes]

    def get_nodes_features(self, nodes: typing.Union[list, np.ndarray, None]) -> np.ndarray:
        if nodes is None:
            return self._node_feature
        else:
            return self._node_feature[nodes]
