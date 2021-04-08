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
        self.embedding = keras.layers.Embedding(input_dim=self.word_size, output_dim=self.embed_size,
                                                embeddings_initializer="he_uniform",
                                                embeddings_regularizer="l2")
        self.loss_layer = SampleSoftmaxLoss(self.word_size, self.num_sampled)

        self.built = True

    def call(self, inputs, training=None, mask=None):
        if training:
            inp, tar = inputs
            inp = tf.reshape(inp, (-1,))
            embed = self.embedding(inp)
            self.loss_layer((tar, embed))
            return embed
        else:
            inp = tf.reshape(inputs, (-1))
            embed = self.embedding(inp)
            return embed
