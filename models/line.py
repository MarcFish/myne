import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn import metrics

from .model import Model
from ..graph import StaticGraph


class LINE(Model):
    def __init__(self, graph, embed_size=128,
                 batch=200, epochs=2000, lr=0.001,
                 negative_ratio=5, order=2, p=0.75):
        self.g = graph
        self.embed_size = embed_size
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.neg_ratio = negative_ratio
        self.order = order
        self.neg_p = p

        self.optimizer = keras.optimizers.Adam(self.lr)

        self.embeddings = keras.layers.Embedding(input_dim=len(self.g.nodes()), output_dim=self.embed_size)
        self.context_embeddings = keras.layers.Embedding(input_dim=len(self.g.nodes()), output_dim=self.embed_size)

        self.embedding_matrix = None
        self.reg = None

    def train(self):
        for a, b, sign in self._get_batch():
            loss = self.train_step(a, b, sign)
            print('Loss {:.4f}'.format(loss))

    def test(self):
        pre = metrics.precision_score(self.g.get_adj(), self.reg.get_adj())
        print("precision:{.4f}".format(pre))
        return pre

    def get_embedding_node(self, node):
        if node not in self.g.nodes():
            raise KeyError
        return self.embeddings(self.g.nodes_map[node])

    def get_reconstruct_graph(self, th=0.9):
        if self.reg is None:
            a_new = tf.cast(tf.math.greater(tf.linalg.matmul(self.get_embedding_matrix(), self.get_embedding_matrix(),
                                                             transpose_b=True), th), tf.int32).numpy()
            self.reg = StaticGraph(nx.from_numpy_matrix(a_new))
        return self.reg

    def get_embedding_matrix(self):
        if self.embedding_matrix is None:
            self.embedding_matrix = self.embeddings.get_weights()[0]
        return self.embedding_matrix

    def loss(self, a, b, sign):
        return -tf.math.reduce_mean(tf.math.log_sigmoid(
            sign*tf.reduce_sum(tf.math.multiply(a, b), axis=1)))

    @tf.function
    def train_step(self, a, b, sign):
        if self.order == 1:
            with tf.GradientTape() as tape:
                a_ = self.embeddings(a)
                b_ = self.embeddings(b)
                loss = self.loss(a_, b_, sign)
            gradients = tape.gradient(loss, self.embeddings.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.embeddings.trainable_variables))
            return loss
        elif self.order == 2:
            with tf.GradientTape() as tape, tf.GradientTape() as ctape:
                a_ = self.embeddings(a)
                b_ = self.context_embeddings(b)
                loss = self.loss(a_, b_, sign)
            gradients = tape.gradient(loss, self.embeddings.trainable_variables)
            cgradients = ctape.gradient(loss, self.context_embeddings.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, self.embeddings.trainable_variables))
            self.optimizer.apply_gradients(zip(cgradients, self.context_embeddings.trainable_variables))

            return loss
        else:
            pass  # TODO

    def _get_batch(self):
        mod_ = 0
        mod_size = self.neg_ratio + 1
        edge_list = np.asarray(list(self.g.edge_node_map))
        edge_sample_list = list(range(len(edge_list)))
        edge_weights = np.asarray(self.g.get_edge_weight_list())
        edge_prob = edge_weights/np.sum(edge_weights)

        node_list = self.g.nodes_map()
        node_degrees = np.asarray(self.g.get_node_degree_list())
        node_prob = np.power(node_degrees, self.neg_p)
        node_prob = node_prob/np.sum(node_prob)

        a = list()
        b = list()
        sign = 0

        for epoch in range(self.epochs):
            if mod_ == 0:
                sign = 1.0
                edge_batch = np.random.choice(a=edge_sample_list, p=edge_prob, size=self.batch)
                edge_batch = edge_list[edge_batch]
                a = edge_batch[:, 0]
                b = edge_batch[:, 1]
            else:
                sign = -1.0
                node_batch = np.random.choice(a=node_list, p=node_prob, size=self.batch)
                b = node_batch

            yield a, b, sign
            mod_ += 1
            mod_ %= mod_size


