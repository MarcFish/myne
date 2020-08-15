import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn import metrics

from .model import Model
from ..graph import StaticGraph
from ..utils import train_test_split


class LINE(Model):
    def __init__(self, graph, embed_size=128,
                 batch=1000, epochs=2000, lr=0.01,
                 negative_ratio=5, order=2, p=0.75):
        self.g, self.test_dict = train_test_split(graph)
        self.embed_size = embed_size
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.neg_ratio = negative_ratio
        self.order = order
        self.neg_p = p
        self.node_size = self.g.get_nodes_number()

        self.optimizer = keras.optimizers.Nadam(self.lr)

        self.embeddings = keras.layers.Embedding(input_dim=self.node_size, output_dim=self.embed_size)
        self.context_embeddings = keras.layers.Embedding(input_dim=self.node_size, output_dim=self.embed_size)

        self.embedding_matrix = None
        self.reg = None

    def train(self):
        for a, b, sign in self._get_batch():
            loss = self.train_step(a, b, sign)
            print('Loss {:.4f}'.format(tf.reduce_mean(loss)))

        self.get_embedding_matrix()
        # self.get_reconstruct_graph()

    def test(self):
        pre = metrics.precision_score(self.g.get_adj(), self.reg.get_adj(), average="macro")
        print("precision:{.4f}".format(pre))
        return pre

    def link_pre(self, k=5):
        hit = 0
        recall = 0
        precision = k * len(self.test_dict)
        cand = list()
        for _, v in self.test_dict.items():
            cand.extend(v)
        cand = np.asarray(cand)
        cand_embed = self.embeddings(cand)
        for node,neighbors in self.test_dict.items():
            neighbors = np.asarray(neighbors)
            node_embed = tf.reshape(self.get_embedding_node(node), (1, self.embed_size))
            pre = tf.math.sigmoid(tf.matmul(node_embed, cand_embed, transpose_b=True)).numpy()
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

    def get_embedding_node(self, node, is_map=True):
        if is_map:
            return self.embeddings(node)

    def get_reconstruct_graph(self, th=0.9):
        a_new = tf.cast(tf.math.greater(tf.math.sigmoid(
                tf.linalg.matmul(self.get_embedding_matrix(), self.get_embedding_matrix(), transpose_b=True))
                , th), tf.int32).numpy()
        self.reg = StaticGraph(nx.from_numpy_matrix(a_new))
        return self.reg

    def get_embedding_matrix(self):
        self.embedding_matrix = self.embeddings.get_weights()[0]
        return self.embedding_matrix

    def loss(self, a, b, sign):
        return -tf.math.log_sigmoid(
            sign*tf.reduce_sum(tf.math.multiply(a, b), axis=1))

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
        edge_list = np.asarray(self.g.get_edges_list())
        edge_sample_list = list(range(len(edge_list)))

        node_list = self.g.get_nodes_map_list()
        node_degrees = np.asarray(self.g.get_nodes_degree_list())
        node_prob = np.power(node_degrees, self.neg_p)
        node_prob = node_prob/np.sum(node_prob)

        a = list()
        b = list()
        sign = 0

        for epoch in range(self.epochs):
            if mod_ == 0:
                sign = 1.0
                edge_batch = np.random.choice(a=edge_sample_list, size=self.batch)
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


