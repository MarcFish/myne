import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activations="leakyrelu", dropout_prob=0.1):
        super(DenseLayer, self).__init__()
        self.layers = keras.Sequential()
        for unit in units:
            self.layers.add(keras.layers.Dense(units=unit))
            self.layers.add(keras.layers.LeakyReLU(0.2))
            self.layers.add(keras.layers.DropOut(dropout_prob))
            self.layers.add(keras.layers.LayerNormalization())

    def call(self, inputs):
        o = self.layers(inputs)
        return o


class ResidualLayer(keras.layers.Layer):
    def __init__(self, unit1s, unit2s):
        super(ResidualLayer, self).__init__()
        self.layer1 = keras.Sequential()
        self.layer2 = keras.Sequential()
        self.unit1s = unit1s
        self.unit2s = unit2s
        for unit in unit1s:
            self.layer1.add(keras.layers.Dense(units=unit))
            self.layer1.add(keras.layers.LeakyReLU(0.2))
            self.layer1.add(keras.layers.BatchNormalization())
        for unit in unit2s:
            self.layer2.add(keras.layers.Dense(units=unit))
            self.layer2.add(keras.layers.LeakyReLU(0.2))
            self.layer2.add(keras.layers.BatchNormalization())
        self.leakyrelu = keras.layers.LeakyReLU(0.2)
        self.bn = keras.layers.BatchNormalization()

    def build(self, input_shape):
        if input_shape[-1]!= self.unit2s[-1]:
            raise Exception("Dim not equal")
        self.built = True

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        outputs = self.leakyrelu(x + inputs)
        outputs = self.bn(outputs)
        return outputs
