from NE.data import Cora
from NE.models import LINE
from NE.utils import embed_visual, svm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--l2", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--batch", type=int, default=512)

arg = parser.parse_args()

cora = Cora()
line = LINE(cora.g, embed_size=arg.embed_size, lr=arg.lr, l2=arg.l2,
            batch_size=arg.batch, epochs=arg.epoch, num_sampled=50,  order=2)
line.train()
embed_visual(line.embedding_matrix, cora.label_matrix)
c = svm(line.embedding_matrix, cora.label_matrix)
