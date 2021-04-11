from graph import StaticGraph, TemporalGraph
from copy import deepcopy
from utils import *
from pathlib import Path


class DBLP:
    def __init__(self, path="/home/marcfish/Documents/project/myne/data/dblp/dblp_20.csv", prob=0.7):
        path = Path(path)
        self.g = StaticGraph()
        self.g.read_edge(path)

        self.train_g = StaticGraph()
        self.test_g = StaticGraph()
        self._split(prob)

    def _split(self, prob=0.7):
        adj = deepcopy(self.g.adj_csr).tolil()
        for n in self.g.node_array:
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
    def __init__(self, edge_file="/home/marcfish/Documents/project/myne/data/cora/cora.cites", feature_file="/home/marcfish/Documents/project/myne/data/cora/cora.content", prob=0.7):
        edge_path = Path(edge_file)
        feature_path = Path(feature_file)

        self.g = StaticGraph()
        self.g.read_edge(edge_path)

        self.train_g = StaticGraph()
        self.test_g = StaticGraph()
        feature_dict = dict()
        label_dict = dict()
        for row in read_txt(feature_path):
            node = row[0]
            label = row[-1]
            f = [int(w) for w in row[1:-1]]
            feature = np.asarray(f, dtype=np.float32)
            label_dict[node] = label
            feature_dict[node] = feature
        self.g.read_node_label(label_dict)
        self.g.read_node_feature(feature_dict)
        self._split(prob)

    def _split(self, prob=0.7):
        adj = deepcopy(self.g.adj_csr).tolil()
        for n in self.g.node_array:
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
    def __init__(self, edge_file="/home/marcfish/Documents/project/myne/data/amazon_book/edge.csv", feature_file="/home/marcfish/Documents/project/myne/data/amazon_book/feature.csv", prob=0.7):
        edge_path = Path(edge_file)
        feature_path = Path(feature_file)

        self.g = TemporalGraph()
        self.g.read_edge(edge_path)
        label_dict = dict()
        for row in read_csv(feature_path):
            node = row[0]
            label = row[-1]
            label_dict[node] = label
        self.g.read_node_label(label_dict)

    def _split(self, prob=0.7):
        # TODO
        pass


if __name__ == "__main__":
    dblp = DBLP()
    cora = Cora()
    book = Book()


#
#
# class Math:
#     def __init__(self, path="E:/project/NE/data/mathoverflow/mathoverflow_10.csv", prob=0.7):
#         self.g = TemporalGraph()
#         self.g.read_from_file(path)
#
#         self.train_g = TemporalGraph()
#         self.test_g = TemporalGraph()
#         self._split(prob)
#         self.disc_gs = list()
#
#     def discrete(self, slot=10):
#         self.disc_gs = list()
#         nonzero = self.g.adj_csr[self.g.adj_csr.nonzero()]
#         min_ = np.min(nonzero)
#         max_ = np.max(nonzero)
#         slice_ = (max_ - min_) // slot
#         adj = self.g.adj_csr.toarray()
#         for i in range(1, slot + 1):
#             adj_ = deepcopy(adj)
#             adj_[np.where(adj_ > min_ + slice_ * i)] = 0
#             g = TemporalGraph()
#             g.adj = sp.coo_matrix(adj_)
#             self.disc_gs.append(g)
#
#         return self.disc_gs
#
#     def _split(self, prob=0.7):
#         # TODO time split
#         adj = deepcopy(self.g.adj_csr).tolil()
#         for n in self.g.node_list:
#             delete_num = int(self.g.get_node_degree(n) * (1 - prob))
#             if delete_num == 0:
#                 continue
#             delete_node = np.random.choice(self.g.get_node_neighbors(n), size=delete_num, replace=False)
#             for dn in delete_node:
#                 if self.g.get_node_degree(dn) == 0:
#                     continue
#                 adj[n, dn] = 0
#                 adj[dn, n] = 0
#
#         self.train_g.adj = adj.tocoo()
#         self.test_g.adj = (self.g.adj_csr - self.train_g.adj_csr).tocoo()

