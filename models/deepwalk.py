from gensim.models import Word2Vec
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

from ..walker import BaseWalker
from .model import Model
from ..graph import StaticGraph


class DeepWalk(Model):
    def __init__(self, graph, embed_size=128, window_size=5, iters=5, walk_length=40, num_walks=10, workers=1):
        self.wv = None
        self.g = graph
        self.embeddings = dict()
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

        self.get_reconstruct_graph()
        self.get_embedding_matrix()

    def get_embedding_node(self, node):
        if node not in self.g.get_nodes_list():
            raise KeyError
        return self.wv.wv[node]

    def get_embedding_matrix(self):
        self.embedding_matrix = np.zeros((self.g.get_nodes_number(), self.embed_size))
        for v, i in self.g.get_node_map_iter():
            self.embedding_matrix[i] = self.wv.wv[v]
        return self.embedding_matrix

    def get_reconstruct_graph(self, th=0.9):
        embed = self.get_embedding_matrix()
        embed_sim = cosine_similarity(embed, embed)
        a_new = np.greater(embed_sim, th).astype(np.int32)
        self.reg = StaticGraph()
        self.reg.read_from_array(a_new)
        return self.reg

    def test(self):  # TODO
        pre = metrics.precision_score(self.g.get_adj(), self.reg.get_adj(), average="macro")
        print("precision:{:.4f}".format(pre))
        return pre

    def visual(self):  # TODO
        pass
