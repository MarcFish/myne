from NE.data import Cora
from NE.models import DeepWalk
from NE.utils import embed_visual, svm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--l2", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--batch", type=int, default=2048)

arg = parser.parse_args()

cora = Cora()
dp = DeepWalk(cora.g, embed_size=arg.embed_size, lr=arg.lr, l2=arg.l2, batch_size=arg.batch)
dp.train()
embed_visual(dp.embedding_matrix, cora.label_matrix)
c = svm(dp.embedding_matrix, cora.label_matrix)
