import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_addons as tfa
from .model import Model
from ..layers import GCRN1Cell, GCRN2Cell


class GCRN(keras.Model):
    def __init__(self, node_size, embed_size, label_size, mode=2, stack=1):
        super(GCRN, self).__init__()
        self.node_size = node_size
        self.embed_size = embed_size
        self.mode = mode
        self.stack = stack
        self.label_size = label_size

    def build(self, input_shape):
        seq_len = input_shape[1]
        self.embedding = self.add_weight(shape=(self.node_size, seq_len, self.embed_size),
                                         initializer=keras.initializers.HeUniform())
        if self.mode == 1:
            cell = GCRN1Cell
        else:
            cell = GCRN2Cell
        if self.stack == 1:
            self.rnn = keras.layers.RNN(cell(self.embed_size))
        else:
            cell_list = [cell(self.embed_size) for _ in range(self.stack)]
            stack_cell = keras.layers.StackedRNNCells(cell_list)
            self.rnn = keras.layers.RNN(stack_cell)
        self.dense = keras.layers.Dense(self.label_size, activation="sigmoid")

    def call(self, inputs):  # A
        X = self.embedding
        A = inputs
        input = tf.concat([X, A], axis=-1)
        output = self.rnn(input)
        output = self.dense(output)
        return output
