import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from .model import Model
from ..layers import DenseLayer


class SDNE(Model):
    def __init__(self, graph,
                 embed_size=128, alpha=0.3, beta=10.0,
                 epochs=2000, batch=200, lr=0.001, layer_size_list=None):
        self.g = graph
        self.A = self.g.adj.todense().astype(np.float32)  # TODO: sparse represent
        self.node_size = self.g.node_size

        self.embed_size = embed_size
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.batch = batch
        self.lr = lr

        if layer_size_list is None:
            self.layer_size_list = [32, 64, 128, self.embed_size]
        else:
            self.layer_size_list = layer_size_list

        self.model = _SDNE(self.node_size, self.layer_size_list)
        self.optimizer = keras.optimizers.Adam(self.lr)

        self._embedding_matrix = None

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    def train(self):
        for epoch in range(self.epochs):
            index = np.random.randint(self.node_size, size=self.batch)
            adj_batch_train = self.A[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = np.ones_like(adj_batch_train)
            b_mat_train[adj_batch_train != 0] = self.beta
            loss = self.train_step(adj_batch_train, adj_mat_train, b_mat_train)

            if epoch % 200 == 0:
                print('Epoch {} Loss {:.4f}'.format(epoch, loss))
        self.get_embedding_matrix()

    def similarity(self, x, y):
        x_embed = tf.reshape(self.get_embedding_node(x), (1, self.embed_size))
        y_embed = tf.reshape(self.get_embedding_node(y), (1, self.embed_size))
        return tf.math.sigmoid(tf.matmul(x_embed, y_embed, transpose_b=True)).numpy()

    def get_embedding_matrix(self):  # TODO
        self._embedding_matrix = np.zeros((self.g.node_number, self.g.node_number))
        for v, i in self.g.node_list:
            self._embedding_matrix[i] = self.model(self.A[i])[0].numpy()
        return self._embedding_matrix

    def get_embedding_node(self, node):  # TODO
        embed = self.model(self.A[node])[0].numpy()
        return embed

    def loss_1(self, A, enc_out):
        D = tf.linalg.diag(tf.reduce_sum(A, 1))
        L = D - A
        return 2 * tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(enc_out), L), enc_out))

    def loss_2(self, inp, oup, B):
        return tf.reduce_sum(tf.square((inp - oup) * B))

    def loss_final(self, A, inp, enc_out, dec_out, B):
        return self.loss_2(inp, dec_out, B) + self.alpha * self.loss_1(A, enc_out)

    @tf.function
    def train_step(self, adj_batch_train, adj_mat_train, b_mat_train):
        with tf.GradientTape() as tape:
            enc_out, dec_out = self.model(adj_batch_train)

            loss = self.loss_final(adj_mat_train, adj_batch_train, enc_out, dec_out, b_mat_train)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


class _SDNE(keras.Model):
    def __init__(self, node_size, layer_size_list):
        super(_SDNE, self).__init__()
        layer_size_list.insert(0, node_size)
        self.encoder = DenseLayer(layer_size_list)
        self.decoder = DenseLayer(layer_size_list)

    def call(self, inp):
        enc_output = self.encoder(inp)
        dec_output = self.decoder(enc_output)

        return enc_output, dec_output


