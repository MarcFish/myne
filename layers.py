import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class FMLayer(keras.layers.Layer):
    def __init__(self, k=10, w_lr=1e-2, v_lr=1e-2):
        super(FMLayer, self).__init__()
        self.k = k
        self.w_lr = w_lr
        self.v_lr = v_lr

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer='he_uniform',
                                 regularizer=keras.regularizers.l2(self.w_lr),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
                                 initializer='he_uniform',
                                 regularizer=keras.regularizers.l2(self.v_lr),
                                 trainable=True)

    def call(self, inputs):
        first_order = self.w0 + tf.matmul(inputs, self.W)
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)),
            axis=1, keepdims=True)

        return tf.squeeze(first_order + second_order)


class ResidualLayer(keras.layers.Layer):
    def __init__(self, unit1s, unit2s):
        super(ResidualLayer, self).__init__()
        self.layer1 = keras.Sequential()
        self.layer2 = keras.Sequential()
        self.unit1s = unit1s
        self.unit2s = unit2s
        if type(unit1s) == list:
            for unit in unit1s:
                self.layer1.add(keras.layers.Dense(units=unit))
                self.layer1.add(keras.layers.LeakyReLU(0.2))
                self.layer1.add(keras.layers.BatchNormalization())
        else:
            self.layer1.add(keras.layers.Dense(units=unit1s))
            self.layer1.add(keras.layers.LeakyReLU(0.2))
            self.layer1.add(keras.layers.BatchNormalization())
        if type(unit2s) == list:
            for unit in unit2s:
                self.layer2.add(keras.layers.Dense(units=unit))
                self.layer2.add(keras.layers.LeakyReLU(0.2))
                self.layer2.add(keras.layers.BatchNormalization())
        else:
            self.layer2.add(keras.layers.Dense(units=unit2s))
            self.layer2.add(keras.layers.LeakyReLU(0.2))
            self.layer2.add(keras.layers.BatchNormalization())
        self.leakyrelu = keras.layers.LeakyReLU(0.2)
        self.bn = keras.layers.BatchNormalization()

    def build(self, input_shape):
        if type(self.unit2s)==list:
            if input_shape[-1]!=self.unit2s[-1]:
                raise Exception("Dim not equal")
        else:
            if input_shape[-1]!= self.unit2s:
                raise Exception("Dim not equal")
        self.built = True

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        outputs = self.leakyrelu(x + inputs)
        outputs = self.bn(outputs)
        return outputs


class DenseLayer(keras.layers.Layer):
    def __init__(self, layer_size_list):
        super(DenseLayer, self).__init__()
        self.layers = keras.Sequential()
        for unit in layer_size_list:
            self.layers.add(keras.layers.Dense(units=unit))
            self.layers.add(keras.layers.LeakyReLU(0.2))
            self.layers.add(keras.layers.BatchNormalization())

    def call(self, inputs):
        o = self.layers(inputs)
        return o
