import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_addons as tfa


class LINE(keras.Model):
    def __init__(self, node_size, embed_size, num_sampled=5, order=1):
        super(LINE, self).__init__()
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
        super(LINE, self).build(input_shape)

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

    def call(self, inputs, training=None, mask=None):
        # when fit, shape of embed will be batch, 1, embed_size, but when call
        # shape of embed will be batch, embed_size
        # TODO
        embed = tf.squeeze(tf.nn.embedding_lookup(self.embedding, inputs))
        return embed
