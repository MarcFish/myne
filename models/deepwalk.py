from gensim.models import Word2Vec
import numpy as np
import networkx as nx

from ..walker import BaseWalker
from .model import Model
from ..graph import StaticGraph


class DeepWalk(Model):
    def __init__(self, graph, embed_size=128, window_size=5, iters=5, walk_length=40, num_walks=10, workers=1):
        self.g = graph
        self.wv = None
        # self.embeddings = dict()
        self.embedding_matrix = None
        self.walker = BaseWalker(graph, num_walks, walk_length, workers)
        self.sentences = self.walker.simulate_walks()
        self.embed_size = embed_size
        self.window_size = window_size
        self.iters = iters
        self.workers = workers

        self.reg = None

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

    def get_embedding_node(self, node):
        if node not in self.g.nodes():
            raise KeyError
        return self.wv.wv[node]

    def get_embedding_matrix(self):
        if self.embedding_matrix is None:
            self.embedding_matrix = np.zeros((len(self.g.nodes()), self.embed_size))
            for v, i in self.g.node_map.d.items():
                self.embedding_matrix[i] = self.wv.wv[v]
        return self.embedding_matrix

    def get_reconstruct_graph(self, th=0.9):
        if self.reg is None:
            embed = self.get_embedding_matrix()
            norm = np.linalg.norm(embed, axis=-1)
            norm_ = np.reshape(norm, (norm.shape[0], 1))
            a_new = np.matmul(embed, embed.T)/np.matmul(norm_, norm_.T)
            a_new = np.greater(a_new, th).astype(np.int32)
            self.reg = StaticGraph(nx.from_numpy_matrix(a_new))
        return self.reg

    def test(self):  # TODO
        pass
