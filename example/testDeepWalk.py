import argparse

from NE.data import Cora
from NE.models import DeepWalk
from NE.utils import embed_visual

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--l2", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--batch", type=int, default=2048)

arg = parser.parse_args()

cora = Cora()
dp = DeepWalk(cora.g, epochs=arg.epoch, embed_size=arg.embed_size, lr=arg.lr, l2=arg.l2, batch_size=arg.batch)
dp.train()
embed_visual(dp.embedding_matrix, cora.label_matrix, "../results/img/cora_deepwalk.png")
