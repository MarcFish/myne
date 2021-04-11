import numpy as np
import tensorflow as tf

from graph import StaticGraph, TemporalGraph


class NodeSampler:
    def __init__(self, g: StaticGraph, num_sample: int):
        self.g = g
        self.num_sample = num_sample
        pass


class RandomSubGraph:
    def __init__(self, g: StaticGraph, num_sample: int, num_sample_step: int):
        self.g = g
        self.num_sample = num_sample
        self.num_sample_step = num_sample_step

    def supervised_feature(self):
        def sample():
            for _ in range(self.num_sample_step):
                sampler = np.random.randint(low=0, high=self.g.node_size, size=self.num_sample)
                adjs = self.g.adj_csr[sampler][:, sampler].toarray().astype(np.int32)
                xs = self.g.node_feature[sampler].astype(np.int32)
                ys = self.g.node_label[sampler].astype(np.int32)
                yield (xs, adjs), ys

        sig = ((tf.TensorSpec(shape=(self.num_sample, self.g.node_feature_size), dtype=tf.int32),
                tf.TensorSpec(shape=(self.num_sample, self.num_sample), dtype=tf.int32)),
               tf.TensorSpec(shape=(self.num_sample, ), dtype=tf.int32))
        data = tf.data.Dataset.from_generator(sample, output_signature=sig)
        return data

    def unsupervised_feature(self):
        def sample():
            for _ in range(self.num_sample_step):
                sampler = np.random.randint(low=0, high=self.g.node_size, size=self.num_sample, dtype=np.int32)
                adjs = self.g.adj_csr[sampler][:, sampler].toarray().astype(np.int32)
                xs = self.g.node_feature[sampler].astype(np.int32)
                labels = np.asarray([np.random.choice(self.g.get_node_neighbors(node)) for node in sampler], dtype=np.int32)
                yield (xs, adjs, labels),
        sig = ((tf.TensorSpec(shape=(self.num_sample, self.g.node_feature_size), dtype=tf.int32),
                tf.TensorSpec(shape=(self.num_sample, self.num_sample), dtype=tf.int32),
                tf.TensorSpec(shape=(self.num_sample, ), dtype=tf.int32)),)
        data = tf.data.Dataset.from_generator(sample, output_signature=sig)
        return data


class RandomTemporalSubGraph:
    def __init__(self, g: TemporalGraph, num_sample: int, num_sample_step: int):
        self.g = g
        self.num_sample = num_sample
        self.num_sample_step = num_sample_step

    def supervised(self):
        def sample():
            for _ in range(self.num_sample_step):
                sampler = np.random.randint(low=0, high=self.g.node_size, size=self.num_sample)
                adjs = []
                for adj_csr in self.g.discrete_adj_csr_list:
                    adj = adj_csr[sampler][:, sampler].toarray().astype(np.int32)
                    adjs.append(adj)
                adjs = np.stack(adjs, axis=1)
                ys = self.g.node_label[sampler].astype(np.int32)
                yield (sampler, adjs), ys

        sig = ((tf.TensorSpec(shape=(self.num_sample, ), dtype=tf.int32),
                tf.TensorSpec(shape=(self.num_sample, len(self.g.discrete_adj_list), self.num_sample), dtype=tf.int32)),
               tf.TensorSpec(shape=(self.num_sample, ), dtype=tf.int32))
        data = tf.data.Dataset.from_generator(sample, output_signature=sig)
        return data

    def unsupervised(self):
        def sample():
            for _ in range(self.num_sample_step):
                sampler = np.random.randint(low=0, high=self.g.node_size, size=self.num_sample)
                adjs = []
                for adj_csr in self.g.discrete_adj_csr_list[:-1]:
                    adj = adj_csr[sampler][:, sampler].toarray().astype(np.int32)
                    adjs.append(adj)
                adjs = np.stack(adjs, axis=1)
                labels = np.asarray([np.random.choice(self.g.discrete_g_list[-1].get_node_neighbors(node)) for node in sampler], dtype=np.int32)
                yield (sampler, adjs, labels),

        sig = ((tf.TensorSpec(shape=(self.num_sample, ), dtype=tf.int32),
                tf.TensorSpec(shape=(self.num_sample, len(self.g.discrete_adj_list) - 1, self.num_sample), dtype=tf.int32),
                tf.TensorSpec(shape=(self.num_sample, ), dtype=tf.int32)),)
        data = tf.data.Dataset.from_generator(sample, output_signature=sig)
        return data


class RandomWalkGraph:
    def __init__(self, g: StaticGraph, num_sample: int):
        self.g = g
        self.num_sample = num_sample
