import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

# from ..graph import DynamicGraph
# from ..graph import StaticGraph
from .model import Model


class HTNE(Model):
    def __init__(self, graph, embed_size=128, batch=2000, epochs=2000, lr=0.001, p=0.75, neg_number=2, hist_number=5):
        self.g = graph
        self.embed_size = embed_size
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.neg_p = p
        self.neg_number = neg_number
        self.hist_number = hist_number

        self.optimizer = keras.optimizers.Adam(self.lr)
        node_size = len(self.g.nodes())

        # node_size, embed_size
        self.embeddings = keras.layers.Embedding(input_dim=node_size,
                                                 output_dim=self.embed_size, embeddings_initializer="he_uniform")
        # node_size, 1
        self.delta = keras.layers.Embedding(input_dim=node_size,
                                            output_dim=1, embeddings_initializer=keras.initializers.Ones())

        self.embedding_matrix = None
        self.reg = None

    def train(self):
        for xs, ys, e_times, hs, h_times, neg_node, h_times_mask in self._get_batch():
            loss = self.train_step(xs, ys, e_times, hs, h_times, neg_node, h_times_mask)
            print("Loss {:.4f}".format(np.sum(loss)))

    def test(self):
        pass

    @tf.function
    def train_step(self, xs, ys, e_times,
                   hs, h_times, neg_node, h_times_mask):
        """
        :param xs: batch,
        :param ys: batch,
        :param e_times: batch,
        :param hs: batch, hist_number
        :param h_times: batch, hist_number
        :param neg_node: batch, neg_number
        :param h_times_mask: batch, hist_number
        :return:(batch,),(batch,neg_number)
        """
        with tf.GradientTape() as tape, tf.GradientTape() as delta_tape:
            x_embed = self.embeddings(xs)  # batch, embed_size
            y_embed = self.embeddings(ys)  # batch, embed_size
            h_embed = self.embeddings(hs)  # batch, hist_number, embed_size
            n_embed = self.embeddings(neg_node)  # batch, neg_number, embed_size

            delta = self.delta(xs)  # batch,1
            d_time = tf.math.abs(tf.expand_dims(e_times, axis=1)-h_times)  # batch, hist_number
            p_mu = -tf.reduce_sum(tf.math.square((x_embed-y_embed)), axis=-1)  # batch,
            alpha = -tf.reduce_sum(tf.math.square((tf.expand_dims(x_embed, axis=1)-h_embed)), axis=-1)  # batch, hist_number
            attn = tf.nn.softmax(alpha, axis=1)  # batch, hist_number
            p_lambda = p_mu + tf.reduce_sum(attn*alpha*tf.math.exp(delta * d_time)*h_times_mask, axis=-1)  # batch,

            n_mu = -tf.reduce_sum(tf.math.square((tf.expand_dims(x_embed, axis=1)-n_embed)), axis=-1)  # batch, neg_number
            # batch, hist_number, neg_number
            n_alpha = -tf.reduce_sum(tf.math.square((tf.expand_dims(h_embed, axis=2)-tf.expand_dims(n_embed, axis=1)))
                                     , axis=-1)
            # batch, neg_number
            n_lambda = n_mu + tf.reduce_sum(tf.expand_dims(attn, axis=2)*n_alpha*tf.expand_dims(tf.math.exp(delta*d_time)*h_times_mask, axis=2), axis=1)

            loss = self.loss(p_lambda, n_lambda)
        gradients = tape.gradient(loss, self.embeddings.trainable_variables)
        delta_gradients = delta_tape.gradient(loss, self.delta.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embeddings.trainable_variables))
        self.optimizer.apply_gradients(zip(delta_gradients, self.delta.trainable_variables))

        return loss

    def loss(self, p_lambda, n_lambda):
        return tf.math.log_sigmoid(p_lambda) - tf.reduce_sum(tf.math.log_sigmoid(n_lambda), axis=1)

    def _get_batch(self):
        edge_list = np.asarray(list(self.g.edge_node_map))
        edge_time_list = np.asarray(self.g.get_edge_time_list())
        edge_sample_list = list(range(len(edge_list)))

        node_list = self.g.nodes_map()
        node_degrees = np.asarray(self.g.get_node_degree_list())
        node_prob = np.power(node_degrees, self.neg_p)
        node_prob = node_prob/np.sum(node_prob)

        for epoch in range(self.epochs):
            edge_batch = np.random.choice(a=edge_sample_list, size=self.batch)
            e_times = edge_time_list[edge_batch]
            edge_batch = edge_list[edge_batch]
            xs = edge_batch[:, 0]
            ys = edge_batch[:, 1]
            h_n = np.zeros((self.batch, self.hist_number))
            h_t = np.zeros((self.batch, self.hist_number))
            h_t_mask = np.zeros((self.batch, self.hist_number))
            for i, x in enumerate(xs):
                neighbors = np.asarray(self.g.get_node_history_neighbors(x))
                if len(neighbors) < self.hist_number:
                    h_n[i][-len(neighbors):] = neighbors[:, 0]
                    h_t[i][-len(neighbors):] = neighbors[:, 1]
                    h_t_mask[i][-len(neighbors):] = 1
                else:
                    h_n[i] = neighbors[-self.hist_number:, 0]
                    h_t[i] = neighbors[-self.hist_number:, 1]
                    h_t_mask[i] = 1

            neg_node = np.random.choice(a=node_list, p=node_prob, size=(self.batch, self.neg_number))

            yield xs, ys, e_times.astype(np.float32), h_n, h_t.astype(np.float32), neg_node, h_t_mask.astype(np.float32)

    def get_embedding_matrix(self):
        pass

    def get_reconstruct_graph(self, th=0.9):
        pass

    def get_embedding_node(self, node):
        if node not in self.g.nodes():
            raise KeyError
        return self.embeddings(self.g.node_map[node])
