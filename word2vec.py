import tensorflow as tf
import tensorflow.keras as keras
from layers import SampleSoftmaxLoss


class Word2Vec(keras.Model):
    def __init__(self, word_size, embed_size, num_sampled=5):
        super(Word2Vec, self).__init__()
        self.word_size = word_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled

    def build(self, input_shape):
        self.embedding = self.add_weight(shape=(self.word_size, self.embed_size), initializer="he_uniform", regularizer="l2")
        self.loss_layer = SampleSoftmaxLoss(self.word_size, self.num_sampled)

        self.built = True

    def call(self, inputs, training=None, mask=None):
        if training:
            inp, tar = inputs
            inp = tf.reshape(inp, (-1,))
            embed = tf.nn.embedding_lookup(self.embedding, inp)
            self.loss_layer((tar, embed))
            return embed
        else:
            inp = tf.reshape(inputs, (-1))
            embed = tf.nn.embedding_lookup(self.embedding, inp)
            return embed
