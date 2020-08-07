from collections import OrderedDict
import csv
from copy import deepcopy
import random
import networkx as nx


class MapDict:
    def __init__(self, node_list):

        self._node_list = node_list

        self.d = OrderedDict()
        self.d_inv = OrderedDict()
        for n in node_list:
            self.d.setdefault(n, len(self.d))
            self.d_inv.setdefault(self.d[n], n)

    def get(self, key):
        if key in self.d:
            return self.d[key]
        elif key in self.d_inv:
            return self.d_inv[key]
        else:
            raise KeyError

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        elif key in self.d_inv:
            return self.d_inv[key]
        else:
            raise KeyError

    def __len__(self):
        if len(self.d) != len(self.d_inv):
            raise Exception("length not equal")
        return len(self.d)

    def iter_node(self):
        """
        return a iter to traverse all node
        :return:
        """
        return self.d.keys()

    def iter_node_map(self):
        """
        return a iter to traverse all node map
        :return:
        """
        return self.d_inv.keys()

    def get_map_list(self, node_list):
        """
        return a list contains node map from node_list
        :param node_list:
        :return:
        """
        return [self.d[node] for node in node_list]

    def get_inv_map_list(self, map_list):
        """
        return a list contains node from map_list
        :param map_list:
        :return:
        """
        return [self.d_inv[node_map] for node_map in map_list]

    def pop(self, key):
        print(key)
        self.d_inv.pop(self.d.get(key))
        self.d.pop(key)


def allocation_num(num, workers):  # TODO: workers=-1
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]


def read_csv(filename, with_header=True):
    with open(filename, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        if with_header:
            header = next(f_csv)
        for row in f_csv:
            yield row


def write_csv(filename, content_iter):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in content_iter:
            writer.writerow(row)


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for row in f:
            yield row.strip()


def train_test_split(sg, th):
    train_sg = deepcopy(sg)
    test_sg = deepcopy(sg)
    for edge in sg._g.edges():
        v0 = edge[0]
        v1 = edge[1]
        v0_map = sg.get_node_map(v0)
        v1_map = sg.get_node_map(v1)
        if random.uniform(0, 1) < th:
            train_sg._g.remove_edge(v0, v1)
            train_sg._map_g.remove_edge(v0_map, v1_map)
            train_sg._edge_map.pop((v0_map, v1_map))
        else:
            test_sg._g.remove_edge(v0, v1)
            test_sg._map_g.remove_edge(v0_map, v1_map)
            test_sg._edge_map.pop((v0_map, v1_map))
    train_sg._map_adj = nx.adjacency_matrix(train_sg._map_g)
    test_sg._map_adj = nx.adjacency_matrix(test_sg._map_g)
