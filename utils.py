from collections import OrderedDict
import collections
import csv
import random
from sklearn.svm import SVC
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf


class MapDict:
    def __init__(self, node_list):

        self._node_list = node_list

        self.d = OrderedDict()
        self.d_inv = OrderedDict()
        for n in node_list:
            self.d.setdefault(n, len(self.d))
            self.d_inv.setdefault(self.d[n], n)

    def get(self, key):
        if key in self.d:
            return self.d[key]
        elif key in self.d_inv:
            return self.d_inv[key]
        else:
            raise KeyError

    def __getitem__(self, key):  # TODO bug
        if key in self.d:
            return self.d[key]
        elif key in self.d_inv:
            return self.d_inv[key]
        else:
            raise KeyError

    def __len__(self):
        if len(self.d) != len(self.d_inv):
            raise Exception("length not equal")
        return len(self.d)

    def iter_node(self):
        """
        return a iter to traverse all node
        :return:
        """
        return self.d.keys()

    def iter_node_map(self):
        """
        return a iter to traverse all node map
        :return:
        """
        return self.d_inv.keys()

    def get_map_list(self, node_list):
        """
        return a list contains node map from node_list
        :param node_list:
        :return:
        """
        return [self.d[node] for node in node_list]

    def get_inv_map_list(self, map_list):
        """
        return a list contains node from map_list
        :param map_list:
        :return:
        """
        return [self.d_inv[node_map] for node_map in map_list]

    def pop(self, key):
        self.d_inv.pop(self.d.get(key))
        self.d.pop(key)


def allocation_num(num, workers):  # TODO: workers=-1
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]


def read_csv(filename, with_header=True):
    with open(filename, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        if with_header:
            header = next(f_csv)
        for row in f_csv:
            yield row


def write_csv(filename, content_iter,append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(filename, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in content_iter:
            writer.writerow(row)


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for row in f:
            yield row.strip()


def scatter2d(x, y):
    plt.scatter(x, y)
    plt.show()
    return


def embed_visual(embedding_matrix, label_array=None, filename=None):
    x_embed = TSNE(n_components=2).fit_transform(embedding_matrix)
    if label_array is None:
        scatter2d(x_embed[:, 0], x_embed[:, 1])
    else:
        label_set = set(label_array)
        plt.figure()
        for label in label_set:
            x = x_embed[np.where(label_array == label)[0]]
            plt.scatter(x[:, 0], x[:, 1], label=label, s=3.0)
            plt.legend()
        plt.show()
    if filename is not None:
        plt.savefig(filename)


def svm(embedding_matrix, label_array):
    c = SVC()
    c.fit(embedding_matrix, label_array)
    print(f"mean accuracy :{c.score(embedding_matrix, label_array)}")
    return c


def generate_word(sentences, num_skips=2, skip_window=2):  # generate train data for word2vec from sentences
    batch = np.ndarray(shape=(len(sentences)*len(sentences[0])*num_skips,), dtype=np.int64)
    label = np.ndarray(shape=(len(sentences)*len(sentences[0])*num_skips, 1), dtype=np.int64)
    span = 2*skip_window + 1
    buffer = collections.deque(maxlen=span)
    for s, sentence in enumerate(sentences):
        data_index = 0
        buffer.clear()
        for _ in range(span):
            buffer.append(sentence[data_index])
            data_index = (data_index + 1) % len(sentence)
        for i in range(len(sentence)):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[s * len(sentence) * num_skips + i * num_skips + j] = buffer[skip_window]
                label[s * len(sentence) * num_skips + i * num_skips + j, 0] = buffer[target]
            buffer.append(sentence[data_index])
            data_index = (data_index + 1) % len(sentence)
    return batch, label


def convert_coo_to_sparse(A):
    indices = np.stack([A.row, A.col])
    values = A.data
    dense_shape = A.shape
    return tf.SparseTensor(indices, values, dense_shape)
