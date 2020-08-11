import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

# from ..graph import DynamicGraph
# from ..graph import StaticGraph
from .model import Model


class HTNE(Model):
    def __init__(self, graph, embed_size=128, batch=1000, epochs=200, lr=0.01, p=0.75, neg_number=5, hist_number=5):
        self.g = graph
        self.embed_size = embed_size
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.neg_p = p
        self.neg_number = neg_number
        self.hist_number = hist_number

        self.optimizer = keras.optimizers.Nadam(self.lr)
        self.node_size = self.g.get_nodes_number()

        # node_size, embed_size
        self.embeddings = keras.layers.Embedding(input_dim=self.node_size,
                                                 output_dim=self.embed_size)
        # node_size, 1
        self.delta = keras.layers.Embedding(input_dim=self.node_size,
                                            output_dim=1)

        self.embedding_matrix = None
        self.reg = None

    def train(self):
        for sign, s, t, edge_times_batch, h_s, h_s_times, h_s_mask in self._get_batch():
            loss = self.train_step(sign, s, t, edge_times_batch, h_s, h_s_times, h_s_mask)
            print("Loss {:.4f}".format(tf.reduce_mean(loss)))

    def test(self):
        pass

    def link_pre(self, k=5):
        hit = 0
        recall = 0
        precision = k*self.node_size
        cand = np.asarray(self.g.get_nodes_map_list())
        cand_embed = self.embeddings(cand)
        for node in self.g.get_nodes_map_list():
            neighbors = np.asarray(self.g.get_node_neighbors(node))[:, 0]
            node_embed = tf.reshape(self.get_embedding_node(node), (1, self.embed_size))
            pre = self.g1(node_embed, cand_embed).numpy()
            pre = cand[np.argsort(pre)].tolist()[-k:]
            for n in neighbors:
                if n in pre:
                    hit += 1
            recall += len(neighbors)
        recall = float(hit)/float(recall)
        precision = float(hit)/float(precision)
        print("recall:{:.4f}".format(recall))
        print("precision:{:.4f}".format(precision))
        return recall, precision

    def g1(self, x, y):
        return -tf.reduce_sum(tf.math.square(x - y), axis=-1)

    def g2(self, x, y):
        return -tf.reduce_sum(tf.math.square(tf.expand_dims(x, axis=1) - y), axis=-1)

    @tf.function
    def train_step(self, sign, s, t, edge_times_batch, h_s, h_s_times, h_s_mask):
        with tf.GradientTape(persistent=True) as tape:
            s_embed = self.embeddings(s)  # batch, embed_size
            t_embed = self.embeddings(t)  # batch, embed_size
            h_s_embed = self.embeddings(h_s)  # batch, hist_number, embed_size

            delta = self.delta(s)  # batch,1
            # d_time = tf.math.abs(tf.expand_dims(edge_times_batch, axis=1)-h_s_times)  # batch, hist_number
            d_time = tf.expand_dims(edge_times_batch, axis=1)-h_s_times
            p_mu = self.g1(s_embed, t_embed)  # batch,
            alpha = self.g2(s_embed, h_s_embed)  # batch, hist_number
            attn = tf.nn.softmax(alpha, axis=1)  # batch, hist_number
            p_lambda = p_mu + tf.reduce_sum(attn*alpha*tf.math.exp(-delta * d_time)*h_s_mask, axis=-1)  # batch,

            loss = self.loss(p_lambda, sign)
        gradients = tape.gradient(loss, self.embeddings.trainable_variables)
        delta_gradients = tape.gradient(loss, self.delta.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embeddings.trainable_variables))
        self.optimizer.apply_gradients(zip(delta_gradients, self.delta.trainable_variables))

        return loss

    def loss(self, p_lambda, sign):
        return -tf.math.log_sigmoid(sign*p_lambda)

    def _get_batch(self):
        mod_ = 0
        mod_size = self.neg_number + 1
        edge_list = np.asarray(self.g.get_edges_list())
        edge_sample_list = list(range(len(edge_list)))

        node_list = self.g.get_nodes_map_list()
        node_degrees = np.asarray(self.g.get_nodes_degree_list())
        node_prob = np.power(node_degrees, self.neg_p)
        node_prob = node_prob/np.sum(node_prob)

        sign = 0.0

        for epoch in range(self.epochs):
            if mod_ == 0:
                sign = 1.0
                edge_batch = np.random.choice(a=edge_sample_list, size=self.batch)
                edge_batch = edge_list[edge_batch]
                edge_times_batch = np.asarray(list(map(self.g.get_edge_time, edge_batch.tolist())))
                s = edge_batch[:, 0]
                t = edge_batch[:, 1]
                h_s = np.zeros((self.batch, self.hist_number))
                h_s_times = np.zeros((self.batch, self.hist_number))
                h_s_mask = np.zeros((self.batch, self.hist_number))
                for i, x in enumerate(s):
                    neighbors = np.asarray(self.g.get_node_neighbors(x))
                    if len(neighbors) < self.hist_number:
                        h_s[i][-len(neighbors):] = neighbors[:, 0]
                        h_s_times[i][-len(neighbors):] = neighbors[:, 1]
                        h_s_mask[i][-len(neighbors):] = 1
                    else:
                        h_s[i] = neighbors[-self.hist_number:, 0]
                        h_s_times[i] = neighbors[-self.hist_number:, 1]
                        h_s_mask[i] = 1
            else:
                sign = -1.0
                t = np.random.choice(a=node_list, p=node_prob, size=self.batch)

            yield sign, s, t, edge_times_batch.astype(np.float32), \
                  h_s.astype(np.int32), h_s_times.astype(np.float32), h_s_mask.astype(np.float32)
            mod_ += 1
            mod_ %= mod_size

    def get_embedding_matrix(self):
        return self.embeddings.get_weights()[0]

    def get_reconstruct_graph(self, th=0.9):
        pass

    def get_embedding_node(self, node):
        return self.embeddings(node)
