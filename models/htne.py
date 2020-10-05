import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from .model import Model
from ..utils import process_dataset


class HTNE(Model):
    def __init__(self, graph, embed_size=128, batch=1000, epochs=200, lr=0.001, p=0.75, neg_number=5, hist_number=5):
        self.g = graph
        self.embed_size = embed_size
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.neg_p = p
        self.neg_number = neg_number
        self.hist_number = hist_number

        self.optimizer = keras.optimizers.Adam(self.lr)
        self.node_size = self.g.node_size

        # node_size, embed_size
        self.embeddings = keras.layers.Embedding(input_dim=self.node_size,
                                                 output_dim=self.embed_size)
        # node_size, 1
        self.delta = keras.layers.Embedding(input_dim=self.node_size,
                                            output_dim=1)

        self._embedding_matrix = None

        self._gen_dataset()

    def train(self):
        for epoch in range(self.epochs):
            for source, target, times, h_s, h_s_times, h_s_mask, nt in self.dataset:
                loss = self.train_step(source, target, times, h_s, h_s_times, h_s_mask, nt)
            print("epoch:{} Loss {:.4f}".format(epoch, tf.reduce_mean(loss)))

        self._embedding_matrix = self.embeddings.get_weights()[0]

    def g1(self, x, y):
        return -tf.reduce_sum(tf.math.square(x - y), axis=-1)

    def similarity(self, x, y):
        x_embed = tf.reshape(self._get_embedding_node(x), (1,self.embed_size))
        y_embed = tf.reshape(self._get_embedding_node(y), (1,self.embed_size))
        return (-tf.reduce_sum(tf.math.square(x_embed-y_embed), axis=-1)).numpy()[0]

    def g2(self, x, y):
        return -tf.reduce_sum(tf.math.square(tf.expand_dims(x, axis=1) - y), axis=-1)

    @tf.function
    def train_step(self, source, target, times, h_s, h_s_times, h_s_mask, nt):
        with tf.GradientTape(persistent=True) as tape:
            s_embed = self.embeddings(source)  # batch, embed_size
            t_embed = self.embeddings(target)  # batch, embed_size
            h_s_embed = self.embeddings(h_s)  # batch, hist_number, embed_size
            nt_embed = self.embeddings(nt)  # batch, neg_number, embed_size

            delta = self.delta(source)  # batch,1
            d_time = tf.expand_dims(times, axis=1)-h_s_times
            p_mu = self.g1(s_embed, t_embed)  # batch,
            alpha = self.g2(s_embed, h_s_embed)  # batch, hist_number
            attn = tf.nn.softmax(alpha, axis=1)  # batch, hist_number
            p_lambda = p_mu + tf.reduce_sum(attn*alpha*tf.math.exp(-delta * d_time)*h_s_mask, axis=-1)  # batch,

            n_mu = self.g2(s_embed, nt_embed)  # batch, neg_number
            n_lambda = tf.reduce_sum(attn*n_mu*tf.math.exp(-delta * d_time), axis=-1)  # batch,

            loss = self.loss(p_lambda, n_lambda)
        gradients = tape.gradient(loss, self.embeddings.trainable_variables)
        delta_gradients = tape.gradient(loss, self.delta.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embeddings.trainable_variables))
        self.optimizer.apply_gradients(zip(delta_gradients, self.delta.trainable_variables))

        return loss

    def loss(self, p_lambda, n_lambda):
        return -tf.math.log_sigmoid(p_lambda) - tf.math.log_sigmoid(-n_lambda)

    def _gen_dataset(self):
        edge_list = self.g.edge_list
        t = edge_list[:, 2]
        scaler = StandardScaler()
        scaler.fit(t.reshape(-1, 1).astype(np.float32))

        node_list = self.g.node_list
        node_degrees = np.asarray(self.g.get_nodes_degree_list())
        node_prob = np.power(node_degrees, self.neg_p)
        self.node_prob = node_prob/np.sum(node_prob)

        source = edge_list[:, 0].astype(np.float32)
        target = edge_list[:, 1].astype(np.float32)
        times = scaler.transform(edge_list[:, 2].reshape(-1, 1).astype(np.float32)).squeeze()
        h_s = np.zeros((len(source), self.hist_number)).astype(np.float32)
        h_s_times = np.zeros((len(source), self.hist_number)).astype(np.float32)
        h_s_mask = np.zeros((len(source), self.hist_number)).astype(np.float32)
        for i, s in enumerate(source):
            neighbors = np.asarray(self.g.get_node_neighbors(s, with_time=True))
            neighbors = np.stack(sorted(neighbors, key=lambda x:x[1]),axis=0)
            for j, temp in enumerate(neighbors):
                if temp[1] > edge_list[i, 2]:
                    break
            if j < self.hist_number:
                if j == 0:
                    h_s[i][0] = neighbors[0, 0]
                    h_s_times[i][0] = scaler.transform(neighbors[0, 1].reshape(-1, 1).astype(np.float32)).squeeze()
                    h_s_mask[i][0] = 1
                else:
                    h_s[i][-j:] = neighbors[-j:, 0]
                    h_s_times[i][-j:] = scaler.transform(neighbors[-j:, 1].reshape(-1, 1).astype(np.float32)).squeeze()
                    h_s_mask[i][-j:] = 1
            else:
                h_s[i] = neighbors[-self.hist_number:, 0]
                h_s_times[i] = scaler.transform(
                    neighbors[-self.hist_number:, 1].reshape(-1, 1).astype(np.float32)).squeeze()
                h_s_mask[i] = 1

        self.dataset = process_dataset(tf.data.Dataset.from_tensor_slices((source, target, times, h_s, h_s_times, h_s_mask)),
                                       self._gen_neg, self.batch)

    def _gen_neg(self, source, target, times, h_s, h_s_times, h_s_mask):
        nt = np.random.choice(a=self.g.node_list, p=self.node_prob, size=self.neg_number).astype(np.float32)
        return source, target, times, h_s, h_s_times, h_s_mask, nt

    def _get_embedding_node(self, node):
        return self.embeddings(node)

    def get_embedding_node(self, node):
        return self.embeddings(node).numpy()

    @property
    def embedding_matrix(self):
        return self._embedding_matrix
