import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def train(self):
        return NotImplementedError

    @abc.abstractmethod
    def test(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_embedding_node(self, node):
        return NotImplementedError

    @abc.abstractmethod
    def get_embedding_matrix(self):
        return NotImplementedError

    @abc.abstractmethod
    def get_reconstruct_graph(self):
        return NotImplementedError

    @abc.abstractmethod
    def visual(self):
        return NotImplementedError
