import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from .model import Model
from ..layers import MeanAggregator


class _UnSupervisedGraphSage(keras.Model):
    def __init__(self, node_size, layer_size_list, concat=False, embed_size=128, feature=None,
                 mode="feature", dropout_prob=0.3, neg_samples=5):
        super(_UnSupervisedGraphSage, self).__init__()
        self.node_size = node_size
        self.embed_size = embed_size
        self.layer_size_list = layer_size_list
        self.hidden = list()
        self.mode = mode
        if self.mode == "feature":
            self.feature = feature
            self.embed_size = feature.shape[-1]
            self.loss = self._loss1
        else:
            self.embed_size = embed_size
            self.loss = self._loss2
        self.embedding = None
        self.dropout_prob = dropout_prob
        self.concat = concat
        self.neg_samples = neg_samples

    def build(self, input_shape):
        if self.mode == "feature":
            self.embedding = tf.constant(self.feature)
        else:
            self.embedding = self.add_weight(shape=(self.node_size, self.embed_size),
                                             initializer=keras.initializers.HeUniform())
        self.biases = self.add_weight(shape=(self.node_size,),
                                      initializer=keras.initializers.zeros())
        out_size = self.embed_size
        for layer_size in self.layer_size_list:
            hidden = MeanAggregator(layer_size, dropout_prob=self.dropout_prob, concat=self.concat)
            hidden.build([(None, out_size), (None, None, self.embed_size)])
            _, out_size = hidden.compute_output_shape(input_shape)
            self.hidden.append(hidden)
        self.nce_weights = self.add_weight(shape=(self.node_size, out_size))

    def call(self, inputs):
        self_vec = tf.squeeze(tf.nn.embedding_lookup(self.embedding, inputs[0]))
        neigh_samples = inputs[1]
        for i, hidden in enumerate(self.hidden):
            neigh_sample = neigh_samples[i]
            neigh_vec = tf.nn.embedding_lookup(self.embedding, neigh_sample)
            self_vec = hidden((self_vec, neigh_vec))

        return self_vec

    def _loss1(self, labels, embed):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.nce_weights,
                                                         biases=self.biases,
                                                         inputs=embed,
                                                         labels=labels,
                                                         num_sampled=self.neg_samples,
                                                         num_classes=self.node_size))
        return loss

    def _loss2(self, labels, embed):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self.nce_weights,
                                                 biases=self.biases,
                                                 inputs=embed,
                                                 labels=labels,
                                                 num_sampled=self.neg_samples,
                                                 num_classes=self.node_size))
        return loss


class UnSupervisedGraphSage(Model):
    def __init__(self, graph, feature=None, mode="feature", feature_size=128, batch_size=200, lr=1e-3, l2=1e-4, dropout_prob=0.3, neigh_samples=25,
                 neg_samples=20, epochs=10, layer_size_list=[128, 256, 512], concat=False):
        self.g = graph
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.dropout_prob = 0.3
        self.neigh_samples = neigh_samples
        self.epochs = epochs
        self.layer_size_list = layer_size_list
        self.neigh_samples = neg_samples
        self.mode = mode
        if mode == "feature":
            self.model = _UnSupervisedGraphSage(self.g.node_size, layer_size_list, concat, feature=feature, mode=mode, dropout_prob=dropout_prob)
        else:
            self.model = _UnSupervisedGraphSage(self.g.node_size, layer_size_list, concat, embed_size=feature_size, mode=mode, dropout_prob=dropout_prob)
        self.model.compile(loss=self.model.loss,
                           optimizer=tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.l2))

        self._embedding_matrix = None

    def train(self):
        batch, neigh_sample_list, label = self._gen_data()
        self.model.fit(x=[batch, neigh_sample_list], y=label, batch_size=self.batch_size, epochs=self.epochs)

        self.get_embedding_matrix()

    def _gen_data(self):
        edge = self.g.edge_list
        batch = edge[:, 0].astype(np.int64)
        label = edge[:, 1].astype(np.int64)
        neigh_sample_list = list()
        for _ in self.layer_size_list:
            neigh_samples = np.ndarray((len(edge), self.neigh_samples), dtype=np.int32)
            for i, node in enumerate(batch):
                neigh = self.g.get_node_neighbors(node)
                neigh_sample = np.random.choice(neigh, size=self.neigh_samples, replace=True)
                neigh_samples[i] = neigh_sample
            neigh_sample_list.append(neigh_samples)
        return batch, neigh_sample_list, label

    def get_embedding_node(self, node):
        return self.embedding_matrix[node]

    def get_embedding_matrix(self):
        batch = self.g.node_list
        neigh_sample_list = list()
        for _ in self.layer_size_list:
            neigh_samples = np.ndarray((self.g.node_size, self.neigh_samples), dtype=np.int32)
            for i, node in enumerate(batch):
                neigh = self.g.get_node_neighbors(node)
                neigh_sample = np.random.choice(neigh, size=self.neigh_samples, replace=True)
                neigh_samples[i] = neigh_sample
            neigh_sample_list.append(neigh_samples)
        self._embedding_matrix = self.model([batch, neigh_sample_list]).numpy()

    @property
    def embedding_matrix(self):
        return self._embedding_matrix


class _SupervisedGraphSage(keras.Model):
    def __init__(self, node_size, layer_size_list, label_size, concat=False, embed_size=None, feature=None,
                 mode="feature", dropout_prob=0.3, activation="sigmoid"):
        super(_SupervisedGraphSage, self).__init__()
        self.node_size = node_size
        self.embed_size = embed_size
        self.layer_size_list = layer_size_list
        self.hidden = list()
        self.mode = mode
        if self.mode == "feature":
            self.feature = feature
            self.embed_size = feature.shape[-1]
        else:
            self.embed_size = embed_size
        self.embedding = None
        self.dropout_prob = dropout_prob
        self.label_size = label_size
        self.activation = keras.activations.get(activation)
        self.concat = concat

    def build(self, input_shape):
        if self.mode == "feature":
            self.embedding = tf.constant(self.feature)
        else:
            self.embedding = self.add_weight(shape=(self.node_size, self.embed_size),
                                             initializer=keras.initializers.HeUniform())
        out_size = self.embed_size
        for layer_size in self.layer_size_list:
            hidden = MeanAggregator(layer_size, dropout_prob=self.dropout_prob, concat=self.concat)
            hidden.build([(None, out_size), (None, None, self.embed_size)])
            _, out_size = hidden.compute_output_shape(input_shape)
            self.hidden.append(hidden)
        self.dense = self.add_weight(shape=(out_size, self.label_size))

    def call(self, inputs):
        self_vec = tf.squeeze(tf.nn.embedding_lookup(self.embedding, inputs[0]))
        neigh_samples = inputs[1]
        for i, hidden in enumerate(self.hidden):
            neigh_sample = neigh_samples[i]
            neigh_vec = tf.nn.embedding_lookup(self.embedding, neigh_sample)
            self_vec = hidden((self_vec, neigh_vec))
        output = tf.matmul(self_vec, self.dense)
        output = self.activation(output)

        return output


class SupervisedGraphSage(Model):
    def __init__(self, graph, label_size, label_matrix, feature=None, mode="feature", feature_size=128, batch_size=200, lr=1e-3,
                 l2=1e-4, dropout_prob=0.3, neigh_samples=25, epochs=10, layer_size_list=[128,128,128], concat=True):
        self.g = graph
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.label_size = label_size
        self.label_matrix = label_matrix
        self.lr = lr
        self.l2 = l2
        self.dropout_prob = 0.3
        self.neigh_samples = neigh_samples
        self.epochs = epochs
        self.layer_size_list = layer_size_list
        if mode == "feature":
            self.model = _SupervisedGraphSage(self.g.node_size, layer_size_list, label_size, concat, feature=feature, mode=mode, dropout_prob=dropout_prob)
        else:
            self.model = _SupervisedGraphSage(self.g.node_size, layer_size_list, label_size, concat, embed_size=feature_size, mode=mode, dropout_prob=dropout_prob)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.l2),
                           metrics=['categorical_accuracy'])

    def train(self):
        batch, neigh_sample_list = self._gen_data()
        label_matrix = tf.one_hot(self.label_matrix, self.label_size)
        self.model.fit(x=[batch, neigh_sample_list], y=label_matrix, batch_size=self.batch_size, epochs=self.epochs)

    def _gen_data(self):
        batch = self.g.node_list
        neigh_sample_list = list()
        for _ in self.layer_size_list:
            neigh_samples = np.ndarray((self.g.node_size, self.neigh_samples),dtype=np.int32)
            for i, node in enumerate(batch):
                neigh = self.g.get_node_neighbors(node)
                neigh_sample = np.random.choice(neigh, size=self.neigh_samples, replace=True)
                neigh_samples[i] = neigh_sample
            neigh_sample_list.append(neigh_samples)
        return batch, neigh_sample_list

    def get_embedding_node(self, node):
        pass

    def get_embedding_matrix(self):
        pass

    @property
    def embedding_matrix(self):
        pass
