import numpy as np
import abc
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Graph(abc.ABC):
    @abc.abstractmethod
    def read_from_file(self, filename, mode):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def adj(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def node_map(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def node_size(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def edge_size(self):
        return NotImplementedError


class StaticGraph(Graph):
    def __init__(self):
        self._adj = None
        self._adj_csr = None
        self._node_map = LabelEncoder()

    @property
    def adj(self):
        return self._adj

    @adj.setter
    def adj(self, x):
        self._adj = x

    @property
    def node_size(self):
        return self._adj.shape[0]

    @property
    def edge_size(self):
        return self._adj.nnz

    @property
    def node_map(self):
        return self._node_map

    @property
    def node_list(self):
        return np.arange(0, self._adj.shape[0])

    @property
    def edge_list(self):
        return np.stack((self._adj.row, self._adj.col), axis=-1)

    def read_from_file(self, filename, file_mode="csv", mode="edge_list"):
        file = pd.read_csv(filename)
        node_set = set(file.values.flatten())
        self._node_map.fit(list(node_set))
        file['x'] = self._node_map.transform(file['x'])  # TODO
        file['y'] = self._node_map.transform(file['y'])
        self._adj = coo_matrix((np.ones(len(file['x'])), (file['x'].values,
                                                            file['y'].values)), shape=(len(node_set), len(node_set)))
        self._adj_csr = self._adj.tocsr()

    def get_node_neighbors(self, node):
        return self._adj_csr[node].indices

    def get_node_degree(self, node):
        return self._adj.csr[node].nnz

    def get_nodes_degree_list(self):
        return np.asarray([self.get_node_degree(n) for n in self.node_list])


class TemporalGraph(StaticGraph):
    def __init__(self):
        super(TemporalGraph, self).__init__()

    @property
    def adj(self):
        return self._adj

    @adj.setter
    def adj(self, x):
        self._adj = x
        self._adj_csr = self._adj.tocsr()

    @property
    def edge_list(self):
        return np.stack((self._adj.row, self._adj.col, self._adj.data), axis=-1)

    def read_from_file(self, filename, file_mode="csv", mode="edge_list"):
        file = pd.read_csv(filename)
        node_set = set(file.values.flatten())
        self._node_map.fit(list(node_set))
        file['x'] = self._node_map.transform(file['x'])  # TODO
        file['y'] = self._node_map.transform(file['y'])
        self._adj = coo_matrix((file['time'].values, (file['x'].values,
                                                            file['y'].values)), shape=(len(node_set), len(node_set)))
        self._adj_csr = self._adj.tocsr()

    def get_node_neighbors(self, node, with_time=False):
        if with_time:
            return np.stack((self._adj_csr[node].indices, self._adj_csr[node].data), axis=-1)
        else:
            return self._adj_csr[node].indices

