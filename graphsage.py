from data import Cora
from layers import GraphSageConv, SampleSoftmaxLoss
from utils import embed_visual, svm
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import argparse
from sampler import RandomSubGraph


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
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--dropout_prob", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_sample_step", type=int, default=100)

arg = parser.parse_args()

cora = Cora()

# supervised
A_in = keras.layers.Input((None, ), batch_size=arg.batch_size)
X_in = keras.layers.Input((cora.g.node_feature_size,), batch_size=arg.batch_size)
o = GraphSageConv(arg.embed_size)([X_in, A_in])
graphsage_ = keras.Model(inputs=[X_in, A_in], outputs=o)
o = keras.layers.Dropout(arg.dropout_prob)(o)
o = GraphSageConv(arg.embed_size)([o, A_in])
o = GraphSageConv(cora.g.label_size, activation="sigmoid")([o, A_in])

supervised_graphsage = keras.Model(inputs=[X_in, A_in], outputs=o)
supervised_graphsage.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay),
            metrics=[keras.metrics.SparseCategoricalAccuracy()])
A = cora.g.adj.toarray()
X = cora.g.node_feature
sample = RandomSubGraph(cora.g, arg.batch_size, arg.num_sample_step)
data = sample.supervised_feature()
supervised_graphsage.fit(data, epochs=arg.epoch, shuffle=False)
embedding_matrix = graphsage_([X, A])
embed_visual(embedding_matrix, cora.g.node_label, filename="./results/img/cora_graphsage_supervised.png")

# unsupervised
A_in = keras.layers.Input((None, ), batch_size=arg.batch_size)
X_in = keras.layers.Input((cora.g.node_feature_size,), batch_size=arg.batch_size)
labels = keras.layers.Input((), batch_size=arg.batch_size)
o = GraphSageConv(arg.embed_size)([X_in, A_in])
o = keras.layers.Dropout(arg.dropout_prob)(o)
o = GraphSageConv(arg.embed_size)([o, A_in])
o = GraphSageConv(arg.embed_size)([o, A_in])
train_o = SampleSoftmaxLoss(node_size=cora.g.node_size)([labels, o])
unsupervised_graphsage = keras.Model(inputs=[X_in, A_in], outputs=o)
unsupervised_graphsage_train = keras.Model(inputs=[X_in, A_in, labels], outputs=train_o)
unsupervised_graphsage_train.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay))

A = cora.g.adj.toarray()
X = cora.g.node_feature
sample = RandomSubGraph(cora.g, arg.batch_size, arg.num_sample_step)
data = sample.unsupervised_feature()
unsupervised_graphsage_train.fit(data, epochs=arg.epoch, shuffle=False)
embedding_matrix = unsupervised_graphsage([X, A])
embed_visual(embedding_matrix, cora.g.node_label, filename="./results/img/cora_graphsage_unsupervised.png")
