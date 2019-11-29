""" Module containing the Differentiable Neural Dictionary, helper classes and
    functions.
"""
import pickle
from collections import deque
import torch

from sklearn.neighbors import KDTree
from xxhash import xxh64 as xxhs


def inverse_distance_kernel(x, xs, delta=0.001):
    """ Computes w_i = k(h, h_i) / sum(h, h_j),
        where k(h, h_i) = 1 / (| h - h_i |^2 + delta)
    """
    distances = 1 / torch.add(torch.norm(xs - x, "fro", 1, keepdim=True), delta)
    return distances / distances.sum()


def _hash(key):
    assert isinstance(key, torch.Tensor), "This key is not a torch.Tensor."
    return xxhs(pickle.dumps(key.numpy())).hexdigest()


class TensorDict:
    """ This might be a mistake :)
    """

    def __init__(self, max_size, key_size, device):
        self.__max_size = max_size
        self.__hash2idx = {}
        self.__keys = torch.zeros(max_size, key_size).to(device)
        self.__values = torch.zeros(max_size, 1).to(device)
        self.__available_idxs = deque(reversed(range(max_size)))

    def keys(self):
        """ Returns the actual tensors representing the keys of the dict.
        """
        return self.__keys

    def values(self):
        """ Returns the values of the dict.
        """
        return self.__values

    def items(self):
        """ Returns an iterator with key, value pairs.
        """
        return zip(self.__keys, self.__values)

    def key2idx(self, key):
        """ Returns the underlying index of a key.
            Used for syncing other data structures.
        """
        return self.__hash2idx[_hash(key)]

    @property
    def is_full(self):
        return len(self.__hash2idx) == self.__max_size

    def __getitem__(self, key):
        return self.__values[self.__hash2idx[_hash(key)]]

    def __setitem__(self, key, value):
        """ Inserts a new key if the dict is not full or it updates an existing
            key. Otherwise it throws a KeyError.
        """
        key_hash = _hash(key)
        if key_hash in self.__hash2idx:
            # updates the value of existing key
            idx = self.__hash2idx[key_hash]
        elif self.__available_idxs:
            # new key, insert into an available slot
            idx = self.__available_idxs.pop()
            self.__hash2idx[key_hash] = idx
        else:
            raise Exception(f"TensorDict is full, delete a key before adding.")
        self.__keys[idx] = key
        self.__values[idx] = value

    def __delitem__(self, key):
        key_hash = _hash(key)
        idx = self.__hash2idx[key_hash]
        # zero-out data
        self.__keys[idx].zero_()
        self.__values[idx].zero_()
        # make the slot available for future inserts
        self.__available_idxs.append(idx)
        del self.__hash2idx[key_hash]

    def __contains__(self, key):
        return _hash(key) in self.__hash2idx

    def __len__(self):
        return len(self.__hash2idx)

    def __str__(self):
        if len(self) == 0:
            return "{}"
        string = ""
        for i, (k, v) in enumerate(self.items()):
            if i == 0:
                string += "{{{}: {},\n".format(str(k), str(v))
            elif i == (len(self) - 1):
                string += " {}: {}}}\n".format(str(k), str(v))
            else:
                string += " {}: {},\n".format(str(k), str(v))
        return string

    def __repr__(self):
        return str(self) + " @ " + str(id(self))


class DND:
    """ Differentiably Neural Dictionary.

        This differentiable data structure maps keys `h` to values `v`, where
        `h` are embeddings (vectors).
    """

    def __init__(self, key_size, device, knn_no=50, max_size=50000):
        self._max_size = max_size
        self._key_size = key_size
        self._knn_no = knn_no
        self._dict = TensorDict(max_size, key_size, device=device)
        self._kd_tree = None  # lazy init
        self._priority = {}
        # self._kde = inverse_distance_kernel
        self._kde = inverse_distance_kernel

    @property
    def ready(self):
        return len(self._dict) > self._knn_no and self._kd_tree is not None

    def write(self, h, v, update_rule=None):
        """ Writes to the DND.
        """
        if h.ndim == 1:
            h = h.unsqueeze(0)
        h = h.data
        if isinstance(v, torch.Tensor):
            v = v.data

        if h in self._dict:
            # old key, update its value
            old_v = self._dict[h]
            self._dict[h] = update_rule(old_v, v)
        elif len(self._dict) < self._max_size:
            # new key, DND not full, append to DND
            self._dict[h] = v
            self._priority[self._dict.key2idx(h)] = 0
        else:
            # new key, DND is full, del least used and append new key to DND
            del self._dict[self._get_least_used()]
            self._dict[h] = v
            self._priority[self._dict.key2idx(h)] = 0

    def lookup(self, h):
        """ Computes the value of `h` based on its closest `K` neighbours.
        """
        # get K nearest neighbours to h
        knn_idxs = self._kd_tree.query(
            h.data.numpy(), k=self._knn_no, return_distance=False
        )
        idxs = torch.from_numpy(knn_idxs).long().squeeze()
        hs = self._dict.keys()[idxs]
        vs = self._dict.values()[idxs]
        # update priorities
        try:
            self._increment_priority(hs)
        except KeyError as kerr:
            print(len(self._dict))
            print(idxs)
            raise kerr

        # compute the weight of each h_i in hs
        weights = self._kde(h, hs)
        # compute the value according to the weights
        return torch.sum(vs * weights, 0, keepdim=True)

    def rebuild_tree(self):
        """ Rebuilds the KDTree using the keys stored so far but only
        if the there are enough keys.
        """
        if self._kd_tree is not None:
            del self._kd_tree

        if self._dict.is_full:
            keys = self._dict.keys().numpy()
        else:
            crt_idx = len(self._dict)
            keys = self._dict.keys().numpy()[:crt_idx]

        if len(self._dict) >= self._knn_no:
            self._kd_tree = KDTree(keys)

    def _increment_priority(self, keys):
        for key in keys:
            self._priority[self._dict.key2idx(key.unsqueeze(0))] += 1

    def _get_least_used(self):
        """ Get the idx of the memory least frequently appearing in nearest
            neighbors searches. I think a smarter data structure is required
            here, maybe a priority queue.
        """
        idx = min(self._priority, key=self._priority.get)
        del self._priority[idx]
        return self._dict.keys()[idx].unsqueeze(0)

    def __len__(self):
        return len(self._dict)

    def __str__(self):
        return "DND(size={size}, key_size={key_size}, K={knn_no})".format(
            size=self._max_size, knn_no=self._knn_no, key_size=self._key_size
        )
