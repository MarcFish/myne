import networkx as nx
import numpy as np
from collections import OrderedDict
import abc
from scipy.sparse import coo_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

from .utils import read_txt, MapDict


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
        self._node_map = LabelEncoder()
        self._neighbor_dict = dict()
        self._edge_list = list()

    @property
    def adj(self):
        return self._adj

    @property
    def node_map(self):
        return self._node_map

    @property
    def neighbor_dict(self):
        return self._neighbor_dict

    def read_from_file(self, filename, file_mode="csv", mode="edge_list"):
        file = pd.read_csv(filename)
        node_set = set(file.values.flatten())
        self._node_map.fit(list(node_set))
        file['node'] = self._node_map.transform(file['node'])  # TODO
        file['node.1'] = self._node_map.transform(file['node.1'])
        self._adj = coo_matrix(np.ones(len(file['node'])), (file['node'].values,
                                                            file['node.1'].values), shape=(len(node_set), len(node_set))).tocsr()

    @property
    def node_list(self):
        return list(self._node_map.iter_node_map())

    @property
    def edge_list(self):
        return self._edge_list

    @property
    def node_number(self):
        return len(self._node_map)

    @property
    def edge_number(self):
        return len(self._edge_list)

    def get_node_neighbors(self, node):
        return list(self._neighbor_dict[node])

    def get_nodes_degree_list(self):
        return [self.get_node_degree(n) for n in self.node_list]

    def get_node_degree(self, node):
        return len(self._neighbor_dict[node])


class TemporalGraph(StaticGraph):
    def __init__(self):
        super(TemporalGraph, self).__init__()

    def read_from_file(self, filename, file_mode="txt", mode="edge_list"):
        node_set = set()
        for row in read_txt(filename):
            row = row.split()
            node_set.add(row[0])
            node_set.add(row[1])
        self._node_map = MapDict(list(node_set))
        self._adj = lil_matrix((len(node_set), len(node_set)))
        for row in read_txt(filename):
            row = row.split()
            v0 = self._node_map[row[0]]
            v1 = self._node_map[row[1]]
            self._adj[v0, v1] = int(row[2])
            self._adj[v1, v0] = int(row[2])
            self._neighbor_dict.setdefault(v0, set())
            self._neighbor_dict.setdefault(v1, set())
            self._neighbor_dict[v0].add((v1, int(row[2])))
            self._neighbor_dict[v1].add((v0, int(row[2])))
            self._edge_list.append([v0, v1, int(row[2])])









