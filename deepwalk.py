import tensorflow_addons as tfa
import argparse
import tensorflow as tf

from walker import BaseWalker
from word2vec import Word2Vec
from utils import generate_word, embed_visual
from data import Cora

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--epoch_size", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=2048)

arg = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


cora = Cora()
walker = BaseWalker(cora.train_g, num_walks=10, walk_length=50, workers=5)
sentences = walker.simulate_walks()
batch, label = generate_word(sentences, num_skips=2, skip_window=3)
model = Word2Vec(cora.g.node_size, arg.embed_size, num_sampled=10)
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay))
model.fit((batch, label), batch_size=arg.batch_size, epochs=arg.epoch_size)

embedding_matrix = model.embedding.numpy()
embed_visual(embedding_matrix, label_array=cora.g.get_nodes_label(), filename="./results/img/cora_deepwalk.png")
