import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_addons as tfa
from .model import Model


class LINE(Model):
    def __init__(self, g, embed_size=128, epochs=10, num_sampled=5, order=1, lr=1e-3, l2=1e-3, batch_size=512):
        self.g = g
        self.model = _LINE(g.node_size, embed_size, num_sampled=num_sampled, order=order)
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        self.epochs = epochs
        self.order = order
        if self.order == 1:
            self.model.compile(loss=self.model.loss1,
                               optimizer=tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.l2))
        else:
            self.model.compile(loss=self.model.lossh,
                               optimizer=tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.l2))

    def train(self):
        batch, label = self._gen_data()
        self.model.fit(batch, label, batch_size=self.batch_size, epochs=self.epochs)

        self.get_embedding_matrix()

    def _gen_data(self):
        edge = self.g.edge_list
        batch = edge[:, 0].astype(np.int64)
        label = edge[:, 1].astype(np.int64)
        return batch, label

    def get_embedding_node(self, node):
        return self._embedding_matrix[node]

    def get_embedding_matrix(self):
        if self.order == 1:
            self._embedding_matrix = self.model(self.g.node_list).numpy()
        else:
            _embedding_matrix = self.model.embedding.numpy()
            _context_embedding_matrix = self.model.context_embedding.numpy()
            self._embedding_matrix = np.concatenate([_embedding_matrix, _context_embedding_matrix], axis=1)
        return self._embedding_matrix

    @property
    def embedding_matrix(self):
        return self._embedding_matrix


class _LINE(keras.Model):
    def __init__(self, node_size, embed_size, num_sampled=5, order=1):
        super(_LINE, self).__init__()
        self.node_size = node_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.order = order

    def build(self, input_shape):
        self.embedding = self.add_weight(shape=(self.node_size, self.embed_size),
                                         initializer=keras.initializers.HeUniform())
        if self.order != 1:
            self.context_embedding = self.add_weight(shape=(self.node_size, self.embed_size),
                                                     initializer=keras.initializers.HeUniform())

        self.biases = self.add_weight(shape=(self.node_size, ), initializer=keras.initializers.zeros())
        self.built = True

    def loss1(self, labels, embed):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.embedding,
                                                 biases=self.biases,
                                                 inputs=embed,
                                                 labels=labels,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.node_size))
        return loss

    def lossh(self, labels, embed):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.context_embedding,
                                                 biases=self.biases,
                                                 inputs=embed,
                                                 labels=labels,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.node_size))
        return loss

    def call(self, inputs):
        # when fit, shape of embed will be batch, 1, embed_size, but when call
        # shape of embed will be batch, embed_size
        # TODO
        embed = tf.squeeze(tf.nn.embedding_lookup(self.embedding, inputs))
        return embed
