import abc
from sklearn.manifold import TSNE
import networkx as nx

from ..utils import scatter2d


class Model(abc.ABC):
    @abc.abstractmethod
    def train(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_embedding_node(self, node):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def embedding_matrix(self):
        return NotImplementedError

    def embed_visual(self):
        x = self.embedding_matrix
        x_embed = TSNE(n_components=2).fit_transform(x)
        plt = scatter2d(x_embed[:, 0], x_embed[:, 1])
        plt.show()
        return plt

    @abc.abstractmethod
    def similarity(self, x, y):
        return NotImplementedError
