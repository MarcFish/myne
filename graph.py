import networkx as nx
import numpy as np
from collections import OrderedDict

from .utils import read_txt, MapDict


class StaticGraph:
    def __init__(self, g=None):
        self._map_g = nx.Graph()
        self._map_adj = None
        self._node_map = None
        self._edge_map = None

        if g is not None:
            self._g = g
            self._node_map = MapDict(list(self._g.nodes))
            for edge in self._g.edges:
                v0 = edge[0]
                v1 = edge[1]
                self._map_g.add_edge(self._node_map.get(v0), self._node_map.get(v1))
            self._edge_map = MapDict(list(self._map_g.edges))
            self._map_adj = nx.adjacency_matrix(self._map_g)
        else:
            self._g = nx.Graph()

    def read_from_edge_list(self, filename):
        for row in read_txt(filename):
            row = row.split()
            self._g.add_edge(row[0], row[1])
        self._node_map = MapDict(list(self._g.nodes))
        for edge in self._g.edges:
            v0 = edge[0]
            v1 = edge[1]
            self._map_g.add_edge(self._node_map.get(v0), self._node_map.get(v1))
        self._edge_map = MapDict(list(self._map_g.edges))
        self._map_adj = nx.adjacency_matrix(self._map_g)

    def read_from_array(self, array):
        self._g = nx.from_numpy_array(array)
        self._node_map = MapDict(list(self._g.nodes))
        for edge in self._g.edges:
            v0 = edge[0]
            v1 = edge[1]
            self._map_g.add_edge(self._node_map.get(v0), self._node_map.get(v1))
        self._edge_map = MapDict(list(self._map_g.edges))
        self._map_adj = nx.adjacency_matrix(self._map_g)

    def get_nodes_list(self):
        return list(self._node_map.iter_node())

    def get_edges_list(self):
        return list(self._edge_map.iter_node())

    def get_nodes_map_list(self):
        return list(self._node_map.iter_node_map())

    def get_edges_map_list(self):
        return list(self._edge_map.iter_node_map())

    def get_node_map_iter(self):
        return self._node_map.d.items()

    def get_edge_map_iter(self):
        return self._edge_map.d.items()

    def get_nodes_number(self):
        return len(self._node_map)

    def get_edges_number(self):
        return len(self._edge_map)

    def get_node_map(self, node):
        return self._node_map[node]

    def get_edge_map(self, edge):
        return self._edge_map[edge]

    def get_node_neighbors(self, v, is_map=True):
        if is_map:
            return list(self._map_g.neighbors(v))
        else:
            return list(self._g.neighbors(v))

    def get_adj_dense(self):
        return np.array(nx.adjacency_matrix(self.g).todense(), dtype='float32')

    def get_adj(self):
        return self._map_adj

    def get_nodes_degree_list(self):
        return [nx.degree(self._map_g, n) for n in self._node_map.iter_node_map()]

    def get_node_degree(self, node):
        return nx.degree(self._g, node)

    def get_node_map_degree(self, node_map):
        return nx.degree(self._map_g, node_map)


class WeightGraph(StaticGraph):
    def __init__(self, g=None):
        super(WeightGraph, self).__init__(g)
        self._weight_list = list()

    def read_from_edge_list(self, filename):
        for row in read_txt(filename):
            row = row.split()
            self._g.add_edge(row[0], row[1], weight=float(row[2]))
        self._node_map = MapDict(list(self._g.nodes))
        for edge in self._g.edges:
            v0 = edge[0]
            v1 = edge[1]
            w = self._g[v0][v1]['weight']
            self._map_g.add_edge(self._node_map.get(v0), self._node_map.get(v1), weight=w)
        self._edge_map = MapDict(list(self._map_g.edges))
        self._map_adj = nx.adjacency_matrix(self._map_g)
        self._weight_list = [self._map_g[edge[0]][edge[1]]['weight']
                             for edge in self._edge_map.iter_node()]

    def get_edge_weight(self, edge):
        return self._weight_list[self._edge_map.get(edge)]

    def get_edge_weight_list(self):
        return self._weight_list

    def get_node_neighbors(self, v, is_map=True):
        return self._map_adj[v].nonzero()[1], \
                   self._map_adj[self._map_adj[v].nonzero()].toarray()[0]


class TemporalGraph(StaticGraph):
    def __init__(self, g=None):
        super(TemporalGraph, self).__init__(g)
        self._temporal_list = list()
        self.neighbors_dict = dict()
        if g is not None:
            for edge in self._g.edges:
                v0 = edge[0]
                v1 = edge[1]
                v0_map = self._node_map[v0]
                v1_map = self._node_map[v1]
                t = self._g[v0][v1]['t']
                self._map_g.add_edge(v0_map, v1_map, t=t)
                self.neighbors_dict.setdefault(v0_map, list())
                self.neighbors_dict.setdefault(v1_map, list())
                self.neighbors_dict[v1_map].append([v0_map, t])
                self.neighbors_dict[v0_map].append([v1_map, t])
            for n, tl in self.neighbors_dict.items():
                tl.sort(key=lambda x: x[1])
            self._temporal_list = [self._map_g[edge[0]][edge[1]]['t'] for edge in self._edge_map.iter_node()]

    def read_from_edge_list(self, filename):
        for row in read_txt(filename):
            row = row.split()
            self._g.add_edge(row[0], row[1], t=float(row[2]))
        self._node_map = MapDict(list(self._g.nodes))
        for edge in self._g.edges:
            v0 = edge[0]
            v1 = edge[1]
            v0_map = self._node_map[v0]
            v1_map = self._node_map[v1]
            t = self._g[v0][v1]['t']
            self._map_g.add_edge(v0_map, v1_map, t=t)
            self.neighbors_dict.setdefault(v0_map, list())
            self.neighbors_dict.setdefault(v1_map, list())
            self.neighbors_dict[v1_map].append([v0_map, t])
            self.neighbors_dict[v0_map].append([v1_map, t])
        for n, tl in self.neighbors_dict.items():
            tl.sort(key=lambda x: x[1])
        self._edge_map = MapDict(list(self._map_g.edges))
        self._map_adj = nx.adjacency_matrix(self._map_g)
        self._temporal_list = [self._map_g[edge[0]][edge[1]]['t'] for edge in self._edge_map.iter_node()]

    def get_edge_time(self, edge):
        return self._temporal_list[self._edge_map.get(tuple(edge))]

    def get_node_neighbors(self, v, is_map=True):
        return self.neighbors_dict.get(v)







