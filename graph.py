import networkx as nx
import abc
import numpy as np
from collections import OrderedDict

from .utils import MapDict


class BaseGraph(abc.ABC):

    @abc.abstractmethod
    def nodes(self):
        return NotImplementedError

    @abc.abstractmethod
    def edges(self):
        return NotImplementedError

    @abc.abstractmethod
    def nodes_map(self):
        return NotImplementedError

    @abc.abstractmethod
    def edges_map(self):
        return NotImplementedError

    @abc.abstractmethod
    def neighbors(self, v):
        return NotImplementedError

    @abc.abstractmethod
    def get_adj_dense(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_adj(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_node_degree_list(self):
        return NotImplementedError


class StaticGraph(BaseGraph):
    def __init__(self, g):
        self.g = g
        self.node_map = MapDict(list(g.nodes))
        self.edge_map = MapDict(list(g.edges))

        edge_list = list()
        for edge in self.edge_map:
            v1 = self.node_map[edge[0]]
            v2 = self.node_map[edge[1]]
            edge_list.append((v1, v2))
        self.edge_node_map = MapDict(edge_list)

    def nodes(self):
        return list(self.node_map)

    def edges(self):
        return list(self.edge_map)

    def nodes_map(self):
        return list(self.node_map.d_inv)

    def edges_map(self):
        return list(self.edge_map.d_inv)

    def neighbors(self, v):
        return list(self.g.neighbors(v))

    def get_adj_dense(self):
        return np.array(nx.adjacency_matrix(self.g).todense(), dtype='float32')

    def get_adj(self):
        return nx.adjacency_matrix(self.g)

    def get_node_degree_list(self):
        degree_list = list()
        for n in self.node_map:
            degree_list.append(nx.degree(self.g, n))
        return degree_list


class WeightGraph(StaticGraph):
    def __init__(self, g):
        super(WeightGraph, self).__init__(g)

        self.edge_weight = OrderedDict()
        for edge in self.edge_map:
            v1 = edge[0]
            v2 = edge[1]
            try:
                weight = self.g[0][1]['weight']
            except KeyError:
                weight = 1
            self.edge_weight[edge] = weight

        self.edge_weight_list = list()
        for edge, weight in self.edge_weight.items():
            self.edge_weight_list.append(weight)

    def get_edge_weight(self, edge):
        return self.edge_weight[edge]

    def get_edge_weight_list(self):
        return self.edge_weight_list


class BiGraph(StaticGraph):
    def __init__(self, edge_list):
        self.a_map = None
        self.b_map = None
        self.g = nx.Graph()

        self.get_graph(edge_list)

        super(BiGraph, self).__init__(g)

    def get_graph(self, edge_list):
        a_list = list()
        b_list = list()
        for a, b in edge_list:
            a_list.append(a)
            b_list.append(b)
            self.g.add_edge(a, b)
        self.a_map = MapDict(a_list)
        self.b_map = MapDict(b_list)

    def nodes_a(self):
        return list(self.a_map)

    def nodes_b(self):
        return list(self.b_map)