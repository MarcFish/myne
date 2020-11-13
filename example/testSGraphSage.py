from NE.models import UnSupervisedGraphSage
from NE.data import Cora
from NE.utils import embed_visual, svm
import argparse
from NE.layers import MeanAggregator
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--l2", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch", type=int, default=512)

arg = parser.parse_args()

cora = Cora()

sgs = UnSupervisedGraphSage(graph=cora.g, feature=cora.feature_matrix,
                          lr=arg.lr, l2=arg.l2, batch_size=arg.batch, epochs=arg.epoch, neigh_samples=10, aggreator="lstm")
sgs.train()
embed_visual(sgs.embedding_matrix, cora.label_matrix)
c = svm(sgs.embedding_matrix, cora.label_matrix)
