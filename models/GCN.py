import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
import argparse
import os

from layers import GraphConvolution, GCNFilter
from data import Cora

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--l2", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--dropout_prob", type=float, default=0.5)

arg = parser.parse_args()

cora = Cora()

X_in = keras.layers.Input(shape=(cora.feature_size,))
A_in = keras.layers.Input(shape=(cora.g.node_size,))
A_o = GCNFilter(mode="others", support=3)(A_in)
o = GraphConvolution(arg.embed_size)([X_in, A_o])
o = keras.layers.Dropout(arg.dropout_prob)(o)
o = GraphConvolution(arg.embed_size)([o, A_o])
o = keras.layers.Dropout(arg.dropout_prob)(o)
o = GraphConvolution(cora.label_size, activation="sigmoid")([o, A_o])

model = keras.Model(inputs=[X_in, A_in], outputs=o)
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.l2),
              loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.Recall()])
model.summary()

X = cora.feature_matrix
A = cora.g.adj.toarray() + np.eye(cora.g.node_size)
Y = tf.one_hot(cora.label_matrix, cora.label_size)
model.fit([X, A], Y, batch_size=cora.g.node_size, epochs=arg.epoch, shuffle=False)
