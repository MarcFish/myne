import collections
import csv
import random
from sklearn.svm import SVC
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf


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
            yield row.strip().split()


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


class Vocab:
    def __init__(self, counter: collections.Counter):
        self.stoi = collections.defaultdict()
        self.itos = list(counter)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def raw_to_seq(self, raw):
        def gen(raw):
            for r in raw:
                for w in r:
                    yield self.stoi[w]
        return list(gen(raw))

    def seq_to_raw(self, seq):
        def gen(seq):
            for s in seq:
                for w in s:
                    yield self.itos[w]
        return list(gen(seq))

    def __len__(self):
        return len(self.itos)
