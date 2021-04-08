import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from data import DBLP


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
        self.built = True

    def _loss1(self, labels, embed):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.embedding,
                                                 biases=self.biases,
                                                 inputs=embed,
                                                 labels=labels,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.node_size))
        return loss

    def _loss2(self, labels, embed):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.context_embedding,
                                                 biases=self.biases,
                                                 inputs=embed,
                                                 labels=labels,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.node_size))
        return loss

    def call(self, inputs, training=None, mask=None):
        if training:
            inp, tar = inputs
            inp = tf.reshape(inp, (-1, ))
            embed = tf.nn.embedding_lookup(self.embedding, inp)
            if self.order == 1:
                loss = self._loss1(tar, embed)
                self.add_loss(loss)
                return embed
            else:
                loss = self._loss2(tar, embed)
                self.add_loss(loss)
                embed2 = tf.nn.embedding_lookup(self.context_embedding, inp)
                return tf.concat([embed, embed2], axis=-1)
        else:
            inp = tf.reshape(inputs, (-1, ))
            embed = tf.nn.embedding_lookup(self.embedding, inp)
            if self.order == 1:
                return embed
            else:
                embed2 = tf.nn.embedding_lookup(self.context_embedding, inp)
                return tf.concat([embed, embed2], axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)

    arg = parser.parse_args()

    dblp = DBLP()
    batch = dblp.g.edge_array[:, 0]
    label = dblp.g.edge_array[:, 1]
    model = LINE(dblp.g.node_size, arg.embed_size, order=2)
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay))
    model.fit((batch, label), batch_size=arg.batch_size, epochs=arg.epoch_size)
