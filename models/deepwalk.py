import tensorflow_addons as tfa

from ..walker import BaseWalker
from .model import Model
from .word2vec import Word2Vec
from ..utils import generate_word


class DeepWalk(Model):
    def __init__(self, graph, embed_size=128, epochs=10, num_sampled=5,
                 walk_length=40, num_walks=10, workers=2, lr=1e-3, l2=1e-2,
                 num_skips=2, skip_window=2, batch_size=256):
        self.g = graph
        self.walker = BaseWalker(graph, num_walks, walk_length, workers)
        self.sentences = self.walker.simulate_walks()
        self.embed_size = embed_size
        self.epochs = epochs
        self.workers = workers
        self.node_size = self.g.node_size
        self.lr = lr
        self.l2 = l2
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.wv = Word2Vec(self.node_size, self.embed_size, num_sampled=num_sampled)
        self.wv.compile(loss=self.wv.loss, optimizer=tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.l2))

        self._embedding_matrix = None

    def train(self):
        batch, label = generate_word(self.sentences, self.num_skips, self.skip_window)
        self.wv.fit(batch, label, batch_size=self.batch_size, epochs=self.epochs)

        self.get_embedding_matrix()

    def get_embedding_node(self, node):
        return self._embedding_matrix[node]

    def get_embedding_matrix(self):
        self._embedding_matrix = self.wv(self.g.node_list).numpy()
        return self._embedding_matrix

    @property
    def embedding_matrix(self):
        return self._embedding_matrix
