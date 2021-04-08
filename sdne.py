import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import argparse
from data import DBLP


class SDNE(keras.Model):
    def __init__(self, node_size, embed_size, alpha=0.3):
        super(SDNE, self).__init__()
        self.node_size = node_size
        self.embed_size = embed_size
        self.alpha = alpha

    def build(self, input_shape):
        self.encoder = keras.Sequential([
            keras.layers.Dense(self.node_size, activation="relu"),
            keras.layers.Dense(self.embed_size, activation="relu"),
            keras.layers.Dense(self.embed_size, activation="relu"),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(self.embed_size, activation="relu"),
            keras.layers.Dense(self.embed_size, activation="relu"),
            keras.layers.Dense(self.node_size, activation="relu"),
        ])

    def _loss_1(self, A, enc_out):
        D = tf.linalg.diag(tf.reduce_sum(A, 1))
        L = D - A
        return 2 * tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(enc_out), L), enc_out))

    def _loss_2(self, inp, oup, B):
        return tf.reduce_sum(tf.square((inp - oup) * B))

    def _loss_final(self, A, inp, enc_out, dec_out, B):
        return self._loss_2(inp, dec_out, B) + self.alpha * self._loss_1(A, enc_out)

    def call(self, inputs, training=None, mask=None):
        if training:
            adj_batch_train, adj_mat_train, b_mat_train = inputs
            enc_output = self.encoder(adj_batch_train)
            dec_output = self.decoder(enc_output)
            loss = self._loss_final(adj_mat_train, adj_batch_train, enc_output, dec_output, b_mat_train)
            self.add_loss(loss)
            return enc_output
        else:
            enc_output = self.encoder(inputs)
            return enc_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("-beta", type=float, default=10.)
    arg = parser.parse_args()

    dblp = DBLP()

    def gen():
        for batch_num in range(dblp.g.node_size // arg.batch_size):
            start_index = batch_num * arg.batch_size
            end_index = (batch_num + 1) * arg.batch_size
            adj_batch_train = dblp.g.adj_csr[start_index:end_index, :].toarray().astype(np.float32)
            adj_mat_train = adj_batch_train[:, start_index:end_index]
            b_mat_train = np.ones_like(adj_batch_train).astype(np.float32)
            b_mat_train[adj_batch_train != 0] = arg.beta
            yield (adj_batch_train, adj_mat_train, b_mat_train),
    data = tf.data.Dataset.from_generator(gen, output_signature=((tf.TensorSpec(shape=[arg.batch_size, dblp.g.node_size], dtype=tf.float32), tf.TensorSpec(shape=[arg.batch_size, arg.batch_size], dtype=tf.float32), tf.TensorSpec(shape=[arg.batch_size, dblp.g.node_size], dtype=tf.float32)),))
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    model = SDNE(dblp.g.node_size, arg.embed_size, alpha=arg.alpha)
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay))
    model.fit(data, epochs=arg.epoch_size)
