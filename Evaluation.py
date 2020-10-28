import numpy as np
from .graph import *
from scipy.sparse import coo_matrix


class Evaluation:
    def __init__(self, g, test_dict=None):
        self.recalls = list()
        self.precisions = list()
        if test_dict is None:
            if type(g) == TemporalGraph:
                data = g.edge_list
                data = np.stack(sorted(data,key=lambda x:x[-1]),axis=0)
                split_ = g.edge_size//10*7
                g.adj = coo_matrix((data[:split_][:,2], (data[:split_][:,0],data[:split_][:,1])), shape=(g.node_size, g.node_size))
                self.test_dict = dict()
                for x,y, _ in data[split_:]:
                    self.test_dict.setdefault(x, list())
                    self.test_dict[x].append(y)
            else:
                pass
        else:
            self.test_dict = test_dict

    def link_pre(self, model, k=5):
        hit = 0
        recall = 0
        precision = k * len(self.test_dict)
        cand = list()
        for _, v in self.test_dict.items():
            cand.extend(v)
        cand = np.asarray(cand)
        for n, neighbors in self.test_dict.items():
            pre = model.similarity(n, cand)
            pre = cand[np.argsort(pre)][-k:]
            hit += len(np.intersect1d(pre, neighbors))
            recall += len(neighbors)
        recall = float(hit)/float(recall)
        precision = float(hit)/float(precision)
        self.recalls.append(recall)
        self.precisions.append(precision)
        print("recall:{:.4f}".format(recall))
        print("precision:{:.4f}".format(precision))
