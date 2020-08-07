import random
import itertools

from joblib import Parallel, delayed

from .utils import allocation_num


class BaseWalker:
    def __init__(self, g, num_walks=10, walk_length=40, workers=1):
        self.g = g
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers

    def simulate_walks(self, verbose=0):

        nodes = list(self.g.get_nodes_list())

        results = Parallel(n_jobs=self.workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num) for num in
            allocation_num(self.num_walks, self.workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.walk_method(start_node=v))
        return walks

    def walk_method(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = self.g.get_node_neighbors(cur,is_map=False)
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk
