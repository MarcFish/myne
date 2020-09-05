import numpy as np


class Evaluation:
    def __init__(self, g, test_dict=None):
        self.recalls = list()
        self.precisions = list()
        if test_dict is None:  # TODO
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
            pre = model.similarity(n, neighbors)
            pre = cand[np.argsort(pre)][-k:]
            hit += len(np.intersect1d(pre, cand))
            recall += len(neighbors)
        recall = float(hit)/float(recall)
        precision = float(hit)/float(precision)
        self.recalls.append(recall)
        self.precisions.append(precision)
        print("recall:{:.4f}".format(recall))
        print("precision:{:.4f}".format(precision))
