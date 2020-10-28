from graph import StaticGraph
import numpy as np
from utils import *
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix


class DBLP:
    def __init__(self, path="./data/dblp/dblp_20.csv", prob=0.7):
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

        self.g.adj = coo_matrix((np.ones(len(xs)), (xs, ys)), shape=(len(node_map.classes_), len(node_map.classes_)))
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
