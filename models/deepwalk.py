from gensim.models import Word2Vec
import numpy as np

from ..walker import BaseWalker
from .model import Model


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
        self.node_size = self.g.node_size

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

    def similarity(self, x, y):
        x_embed = self.get_embedding_node(x).reshape(-1, 1)
        y_embed = self.get_embedding_node(y).reshape(-1, 1)
        return x_embed.dot(y_embed)/(np.linalg.norm(x_embed, ord=2)*np.linalg.norm(y_embed, ord=2))

    def get_embedding_node(self, node):
        return self._embedding_matrix[node]

    def get_embedding_matrix(self):
        self._embedding_matrix = np.zeros((self.node_size, self.embed_size))
        for v in self.g.node_list:
            self._embedding_matrix[v] = self.wv.wv[str(v)]
        return self._embedding_matrix

    @property
    def embedding_matrix(self):
        return self._embedding_matrix
