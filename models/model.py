import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def train(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_embedding_node(self, node):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def embedding_matrix(self):
        return NotImplementedError

    @abc.abstractmethod
    def similarity(self, x, y):
        return NotImplementedError
