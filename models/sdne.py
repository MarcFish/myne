import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .model import Model
from ..graph import StaticGraph
from ..utils import train_test_split


class SDNE(Model):
    def __init__(self, graph,
                 embed_size=128, alpha=0.3, beta=10.0,
                 epochs=2000, batch=200, lr=0.001, layer_size_list=None):
        self.g, self.test_dict = train_test_split(graph)
        self.A = self.g.get_adj().todense().astype(np.float32)
        self.node_size = self.g.get_nodes_number()

        self.embed_size = embed_size
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.batch = batch
        self.lr = lr

        if layer_size_list is None:
            self.layer_size_list = [32, 64, self.embed_size]
        else:
            self.layer_size_list = layer_size_list

        self.model = _SDNE(self.node_size, self.layer_size_list)
        self.optimizer = keras.optimizers.Adam(self.lr)

        self.embeddings = None
        self.reg = None

    def train(self):
        for epoch in range(self.epochs):
            index = np.random.randint(self.node_size, size=self.batch)
            adj_batch_train = self.A[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = np.ones_like(adj_batch_train)
            b_mat_train[adj_batch_train != 0] = self.beta
            loss = self.train_step(adj_batch_train, adj_mat_train, b_mat_train)

            if epoch % 1000 == 0:
                print('Epoch {} Loss {:.4f}'.format(epoch, loss))
        self.get_embedding_matrix()

    def test(self):  # TODO
        pass

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

    def get_embedding_matrix(self):  # TODO
        self.embeddings = np.zeros((len(self.g.nodes()), self.embed_size))
        for v, i in self.g.get_node_map_iter():
            self.embeddings[i] = self.model(self.A[i])[0].numpy()
        return self.embeddings

    def get_reconstruct_graph(self):
        a_new = tf.cast(tf.math.greater(self.model(self.A)[1], 0.9), tf.int32).numpy()
        self.reg = StaticGraph(nx.from_numpy_matrix(a_new))
        return self.reg

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


class _Encoder(keras.layers.Layer):
    def __init__(self, layer_size_list):
        super(_Encoder, self).__init__()
        self.layer_size_list = layer_size_list
        self.layers = list()
        for s in self.layer_size_list:
            self.layers.append(
                keras.layers.Dense(s, activation=keras.activations.relu,
                                                     kernel_regularizer=keras.regularizers.l2))

    def call(self, inp):
        f = inp
        for layer in self.layers:
            f = layer(f)

        return f


class _Decoder(keras.layers.Layer):
    def __init__(self, layer_size_list):
        super(_Decoder, self).__init__()
        self.layer_size_list = list(reversed(layer_size_list))
        self.layers = list()
        for s in self.layer_size_list:
            self.layers.append(
                keras.layers.Dense(s, keras.activations.relu, kernel_regularizer=keras.regularizers.l2))

    def call(self, inp):
        f = inp
        for layer in self.layers:
            f = layer(f)

        return f


class _SDNE(keras.Model):
    def __init__(self, node_size, layer_size_list):
        super(_SDNE, self).__init__()
        layer_size_list.insert(0, node_size)
        self.encoder = _Encoder(layer_size_list)
        self.decoder = _Decoder(layer_size_list)

    def call(self, inp):
        enc_output = self.encoder(inp)
        dec_output = self.decoder(enc_output)

        return enc_output, dec_output


