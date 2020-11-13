from NE.data import Math
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import argparse
from NE.utils import embed_visual, svm

parser = argparse.ArgumentParser()
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--l2", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--dropout_prob", type=float, default=0.3)
parser.add_argument("--slots",type=int, default=10)

arg = parser.parse_args()

m = Math()
