import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class Word2Vec(keras.Model):
    def __init__(self, word_size, embed_size, num_sampled=5):
        super(Word2Vec, self).__init__()
        self.word_size = word_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled

    def build(self, input_shape):
        self.embedding = keras.layers.Embedding(input_dim=self.word_size, output_dim=self.embed_size,
                                                embeddings_initializer="he_uniform",
                                                embeddings_regularizer="l2")
        self.nce_weights = self.add_weight(shape=(self.word_size, self.embed_size),
                                           initializer=keras.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(self.embed_size)))
        self.nce_biases = self.add_weight(shape=(self.word_size, ), initializer=keras.initializers.zeros())

        self.built = True

    def loss(self, labels, embed):
        return tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                                             biases=self.nce_biases,
                                             inputs=embed,
                                             labels=labels,
                                             num_sampled=self.num_sampled,
                                             num_classes=self.word_size))

    def call(self, inputs):
        # when fit embed will is batch, 1, embed_size, but when call
        # embed will is batch, embed_size
        # TODO
        embed = tf.squeeze(self.embedding(inputs))
        return embed
