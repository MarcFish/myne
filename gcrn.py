import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import argparse
import tensorflow_addons as tfa
from layers import GCRN1Cell, GCRN2Cell, GraphAttention, SampleSoftmaxLoss
from data import Book
from sampler import RandomTemporalSubGraph
from utils import embed_visual

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--epoch_size", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("-beta", type=float, default=10.)
parser.add_argument("--num_sample_step", type=int, default=100)
parser.add_argument("--seq_len", type=int, default=10)
arg = parser.parse_args()

book = Book()
book.g.discrete(arg.seq_len)
sample = RandomTemporalSubGraph(book.g, arg.batch_size, arg.num_sample_step)

# supervised
data = sample.supervised()
nodes = keras.layers.Input(shape=(), batch_size=arg.batch_size)
adjs = keras.layers.Input(shape=(arg.seq_len, None), batch_size=arg.batch_size)
nodes_embed = keras.layers.Embedding(input_dim=book.g.node_size, output_dim=arg.embed_size)(nodes)
o = keras.layers.RNN(GCRN2Cell(arg.embed_size, GraphAttention, {"units": arg.embed_size}))(adjs, initial_state=nodes_embed)
cls_o = keras.layers.Dense(book.g.label_size, activation="sigmoid")(o)
supervised_gcrn_ = keras.Model(inputs=(nodes, adjs), outputs=o)
supervised_gcrn = keras.Model(inputs=[nodes, adjs], outputs=cls_o)
supervised_gcrn.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])
supervised_gcrn.fit(data, epochs=arg.epoch_size, shuffle=False)

dis_array = np.stack([adj.toarray() for adj in book.g.discrete_adj_list], axis=1)
embedding_matrix = supervised_gcrn_([book.g.node_array, dis_array])
embed_visual(embedding_matrix, book.g.node_label, filename="./results/img/book_gcrn_supervised.png")

# unsupervised
data = sample.unsupervised()
nodes = keras.layers.Input(shape=(), batch_size=arg.batch_size)
adjs = keras.layers.Input(shape=(None, None), batch_size=arg.batch_size)
labels = keras.layers.Input((), batch_size=arg.batch_size)
nodes_embed = keras.layers.Embedding(input_dim=book.g.node_size, output_dim=arg.embed_size)(nodes)
o = keras.layers.RNN(GCRN2Cell(arg.embed_size, GraphAttention, {"units": arg.embed_size}))(adjs, initial_state=nodes_embed)
o = keras.layers.Dense(book.g.label_size, activation="sigmoid")(o)
train_o = SampleSoftmaxLoss(node_size=book.g.node_size)([labels, o])

unsupervised_gcrn = keras.Model(inputs=[nodes, adjs], outputs=o)
unsupervised_gcrn_train = keras.Model(inputs=[nodes, adjs, labels], outputs=train_o)
unsupervised_gcrn_train.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay))
unsupervised_gcrn_train.fit(data, epochs=arg.epoch_size, shuffle=False)

dis_array = np.stack([adj.toarray() for adj in book.g.discrete_adj_list], axis=1)
embedding_matrix = unsupervised_gcrn([book.g.node_array, dis_array])
embed_visual(embedding_matrix, book.g.node_label, filename="./results/img/book_gcrn_unsupervised.png")
