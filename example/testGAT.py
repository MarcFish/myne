from NE.data import Cora
from NE.layers import GraphAttention
from NE.utils import embed_visual, svm
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--l2", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--dropout_prob", type=float, default=0.3)
parser.add_argument("--batch", type=int, default=512)

arg = parser.parse_args()

cora = Cora()

A_in = keras.layers.Input((cora.g.node_size,))
X_in = keras.layers.Input((cora.feature_size,))
o = GraphAttention(arg.embed_size)([X_in, A_in])
gat_ = keras.Model(inputs=[X_in, A_in], outputs=o)
o = keras.layers.Dropout(arg.dropout_prob)(o)
o = GraphAttention(cora.label_size, attn_heads=1, activation="sigmoid")([o, A_in])

gat = keras.Model(inputs=[X_in, A_in], outputs=o)
gat.compile(loss='categorical_crossentropy',
            optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.l2),
            metrics=['categorical_accuracy'])
A = cora.g.adj.toarray()
X = cora.feature_matrix
Y = tf.one_hot(cora.label_matrix, cora.label_size)
gat.fit([X, A], Y, batch_size=cora.g.node_size, epochs=arg.epoch, shuffle=False)
embedding_matrix = gat_([X, A])
embed_visual(embedding_matrix, cora.label_matrix)
c = svm(embedding_matrix, cora.label_matrix)