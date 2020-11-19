from .graph import StaticGraph, TemporalGraph
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from .utils import *
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from scipy.sparse import coo_matrix


class DBLP:
    def __init__(self, path="E:/project/NE/data/dblp/dblp_20.csv", prob=0.7):
        self.g = StaticGraph()
        self.g.read_from_file(path)

        self.train_g = StaticGraph()
        self.test_g = StaticGraph()
        self._split(prob)

    def _split(self, prob=0.7):
        adj = deepcopy(self.g.adj_csr).tolil()
        for n in self.g.node_list:
            delete_num = int(self.g.get_node_degree(n) * (1 - prob))
            if delete_num == 0:
                continue
            delete_node = np.random.choice(self.g.get_node_neighbors(n), size=delete_num, replace=False)
            for dn in delete_node:
                if self.g.get_node_degree(dn) == 0:
                    continue
                adj[n, dn] = 0
                adj[dn, n] = 0

        self.train_g.adj = adj.tocoo()
        self.test_g.adj = (self.g.adj_csr - self.train_g.adj_csr).tocoo()


class Book:
    def __init__(self, edge_file="E:/project/NE/data/amazon_book/edge.csv",
                 feature_file="E:/project/NE/data/amazon_book/feature.csv", prob=0.7):
        self.g = TemporalGraph()
        self.g.read_from_file(edge_file)

        # read category
        labels = list()
        for row in read_csv(feature_file):
            labels.append(row[1:])
        mlb = MultiLabelBinarizer()
        mlb.fit(labels)
        self.label_size = len(mlb.classes_)
        self.label_matrix = np.ndarray(shape=(self.g.node_size, self.label_size), dtype=np.float32)
        self.label_array = np.ndarray(shape=(self.g.node_size,),dtype=np.int32)
        for row in read_csv(feature_file):
            node = int(row[0])
            labels = mlb.transform([row[1:]])
            self.label_array[node] = int(row[-1])
            self.label_matrix[node] = labels

        self.disc_gs = list()

    def discrete(self, slot=10):
        self.disc_gs = list()
        nonzero = self.g.adj_csr[self.g.adj_csr.nonzero()]
        min_ = np.min(nonzero)
        max_ = np.max(nonzero)
        slice_ = (max_ - min_) // slot
        adj = self.g.adj_csr.toarray()
        for i in range(1, slot + 1):
            adj_ = deepcopy(adj)
            adj_[np.where(adj_ > min_ + slice_ * i)] = 0
            g = TemporalGraph()
            g.adj = sp.coo_matrix(adj_)
            self.disc_gs.append(g)

        return self.disc_gs


class Math:
    def __init__(self, path="E:/project/NE/data/mathoverflow/mathoverflow_10.csv", prob=0.7):
        self.g = TemporalGraph()
        self.g.read_from_file(path)

        self.train_g = TemporalGraph()
        self.test_g = TemporalGraph()
        self._split(prob)
        self.disc_gs = list()

    def discrete(self, slot=10):
        self.disc_gs = list()
        nonzero = self.g.adj_csr[self.g.adj_csr.nonzero()]
        min_ = np.min(nonzero)
        max_ = np.max(nonzero)
        slice_ = (max_ - min_) // slot
        adj = self.g.adj_csr.toarray()
        for i in range(1, slot + 1):
            adj_ = deepcopy(adj)
            adj_[np.where(adj_ > min_ + slice_ * i)] = 0
            g = TemporalGraph()
            g.adj = sp.coo_matrix(adj_)
            self.disc_gs.append(g)

        return self.disc_gs

    def _split(self, prob=0.7):
        # TODO time split
        adj = deepcopy(self.g.adj_csr).tolil()
        for n in self.g.node_list:
            delete_num = int(self.g.get_node_degree(n) * (1 - prob))
            if delete_num == 0:
                continue
            delete_node = np.random.choice(self.g.get_node_neighbors(n), size=delete_num, replace=False)
            for dn in delete_node:
                if self.g.get_node_degree(dn) == 0:
                    continue
                adj[n, dn] = 0
                adj[dn, n] = 0

        self.train_g.adj = adj.tocoo()
        self.test_g.adj = (self.g.adj_csr - self.train_g.adj_csr).tocoo()


class Cora:
    def __init__(self, adj_file="E:/project/NE/data/cora/cora.cites", feature_file="E:/project/NE/data/cora/cora.content", prob=0.7):
        self.g = StaticGraph()
        self.train_g = StaticGraph()
        self.test_g = StaticGraph()
        # read adj file to g
        xs = []
        ys = []
        for row in read_txt(adj_file):
            x, y = row.split()
            xs.append(x)
            ys.append(y)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        nodes = np.concatenate([xs, ys])
        node_map = LabelEncoder()
        node_map.fit(nodes)
        self.g._node_map = node_map
        xs = node_map.transform(xs)
        ys = node_map.transform(ys)
        x = np.concatenate([xs, ys])
        y = np.concatenate([ys, xs])

        adj = coo_matrix((np.ones(len(xs)*2), (x, y)), shape=(len(node_map.classes_), len(node_map.classes_)))
        adj = adj.tocsr() + sp.eye(len(node_map.classes_))
        adj = adj.tocoo()
        self.g.adj = adj
        self._split(prob)

        # read feature
        # row: node, features, label
        self.feature_matrix = np.zeros(shape=(self.g.node_size, 1433))
        self.feature_size = 1433
        self.label_matrix = np.zeros(shape=(self.g.node_size,), dtype=np.int32)
        labels = list()
        for row in read_txt(feature_file):
            row = row.split()
            label = row[-1]
            labels.append(label)
        label_map = LabelEncoder()
        label_map.fit(labels)
        self.label_size = len(label_map.classes_)
        for row in read_txt(feature_file):
            row = row.split()
            node = node_map.transform([int(row[0])])[0]
            feature = np.asarray([int(x) for x in row[1:-1]])
            label = label_map.transform([row[-1]])[0]
            self.feature_matrix[node] = feature
            self.label_matrix[node] = label
        self.feature_matrix = self.feature_matrix / self.feature_matrix.sum(1).reshape(-1, 1)

    def _split(self, prob=0.7):
        adj = deepcopy(self.g.adj_csr).tolil()
        for n in self.g.node_list:
            delete_num = int(self.g.get_node_degree(n) * (1 - prob))
            if delete_num == 0:
                continue
            delete_node = np.random.choice(self.g.get_node_neighbors(n), size=delete_num, replace=False)
            for dn in delete_node:
                if self.g.get_node_degree(dn) == 0:
                    continue
                adj[n, dn] = 0
                adj[dn, n] = 0

        self.train_g.adj = adj.tocoo()
        self.test_g.adj = (self.g.adj_csr - self.train_g.adj_csr).tocoo()
