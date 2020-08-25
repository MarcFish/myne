from gensim.models import Word2Vec
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

from ..walker import BaseWalker
from .model import Model
from ..graph import StaticGraph
from ..utils import train_test_split


class DeepWalk(Model):
    def __init__(self, graph, embed_size=128, window_size=5, iters=5, walk_length=40, num_walks=10, workers=1):
        self.wv = None
        self.g = graph
        self.walker = BaseWalker(graph, num_walks, walk_length, workers)
        self.sentences = self.walker.simulate_walks()
        self.embed_size = embed_size
        self.window_size = window_size
        self.iters = iters
        self.workers = workers
        self.node_size = self.g.node_number

        self._embedding_matrix = None

    def train(self):
        kwargs = dict()
        kwargs["sentences"] = self.sentences
        kwargs["size"] = self.embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 1
        kwargs["workers"] = self.workers
        kwargs["window"] = self.window_size
        kwargs["iter"] = self.iters

        self.wv = Word2Vec(**kwargs)

        self.get_embedding_matrix()

    def get_embedding_node(self, node):
        self._embedding_matrix[node]

    def get_embedding_matrix(self):
        self._embedding_matrix = np.zeros((self.node_size, self.embed_size))
        for v in self.g.node_list:
            self._embedding_matrix[v] = self.wv.wv[str(v)]
        return self._embedding_matrix

    def link_pre(self, test_dict, k=5):
        hit = 0
        recall = 0
        precision = k * self.node_size
        cand = list()
        for _, v in test_dict.items():
            cand.extend(v)
        cand = np.asarray(cand)
        cand_embed = self._embedding_matrix[cand]
        for node, neighbors in test_dict.items():
            neighbors = np.asarray(neighbors)
            node_embed = self.get_embedding_node(node).reshape((1, self.embed_size))
            pre = cosine_similarity(node_embed, cand_embed)
            pre = cand[np.argsort(pre)].tolist()[0][-k:]
            for n in neighbors:
                if n in pre:
                    hit += 1
            recall += len(neighbors)
        recall = float(hit) / float(recall)
        precision = float(hit) / float(precision)
        print("recall:{:.4f}".format(recall))
        print("precision:{:.4f}".format(precision))
        return recall, precision

    @property
    def embedding_matrix(self):
        return self._embedding_matrix



