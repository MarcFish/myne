from collections import OrderedDict


class MapDict:
    def __init__(self, node_list):

        self.node_list = node_list

        self.d = OrderedDict()
        self.d_inv = OrderedDict()
        for n in node_list:
            self.d.setdefault(n, len(self.d))
            self.d_inv.setdefault(self.d[n], n)

    def __getitem__(self, key):
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

    def __iter__(self):
        return iter(self.node_list)


def allocation_num(num, workers):  # TODO: workers=-1
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]