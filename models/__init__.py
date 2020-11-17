from .deepwalk import DeepWalk
from .htne import HTNE
from .sdne import SDNE
from .word2vec import Word2Vec
from .line import LINE
from .graphsage import SupervisedGraphSage, UnSupervisedGraphSage
from .gcrn import GCRN

__all__ = ['DeepWalk', 'HTNE', 'SDNE', 'Word2Vec', "LINE", "SupervisedGraphSage", "UnSupervisedGraphSage", "GCRN"]
