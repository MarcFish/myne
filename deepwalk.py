import tensorflow_addons as tfa
import argparse

from walker import BaseWalker
from word2vec import Word2Vec
from utils import generate_word
from data import DBLP

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--epoch_size", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=2048)

arg = parser.parse_args()

dblp = DBLP()
walker = BaseWalker(dblp.train_g, num_walks=10, walk_length=50, workers=5)
sentences = walker.simulate_walks()
batch, label = generate_word(sentences, num_skips=2, skip_window=3)
model = Word2Vec(dblp.g.node_size, arg.embed_size, num_sampled=10)
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.weight_decay))
model.fit((batch, label), batch_size=arg.batch_size, epochs=arg.epoch_size)
