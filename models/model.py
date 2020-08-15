import abc
from sklearn.manifold import TSNE
import networkx as nx

from ..utils import scatter2d


class Model(abc.ABC):
    @abc.abstractmethod
    def train(self):
        return NotImplementedError

    @abc.abstractmethod
    def test(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_embedding_node(self, node):
        return NotImplementedError

    @abc.abstractmethod
    def get_embedding_matrix(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_reconstruct_graph(self):
        return NotImplementedError

    def embed_visual(self):
        x = self.get_embedding_matrix()
        x_embed = TSNE(n_components=2).fit_transform(x)
        plt = scatter2d(x_embed[:, 0], x_embed[:, 1])
        return plt

    def visual(self):
        nx.draw_spring(self.g._map_g)
        return
