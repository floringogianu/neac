import unittest
from functools import reduce

import torch
from src.dnd import DND


class TestDND(unittest.TestCase):
    """ TestCase for the IOU function. """

    def __init__(self, *args, **kwargs):
        super(TestDND, self).__init__(*args, **kwargs)
        self.precision = 4

    def test_insert_n_different(self):
        """ Insert N different elements in a DND of size N.
        """
        N, key_size = 5, 12
        data = torch.rand(N, key_size)
        dnd = DND(key_size, torch.device("cpu"), max_size=N)

        for i, key in enumerate(data):
            dnd.write(key, i)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

    def test_insert_n_equal(self):
        """ Insert N equal elements in a DND of size N.
        """
        N, key_size = 5, 12
        key = torch.rand(1, key_size)
        data = {key.clone(): val for val in range(N)}

        dnd = DND(key_size, torch.device("cpu"), max_size=N)
        for k, v in data.items():
            dnd.write(k, v, update_rule=lambda old, new: old + new)

        self.assertEqual(len(dnd), 1)  # the DND should be of size one
    
    def test_at_capacity(self):
        """ Insert N+M different elements in a DND of size N.
        """
        N, M, key_size = 5, 7, 12
        data = torch.rand(N+M, key_size)
        dnd = DND(key_size, torch.device("cpu"), max_size=N)

        for i, key in enumerate(data):
            dnd.write(key, i)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

    def test_priority_increment(self):
        """ Insert two clusters and use three cluster elements for a lookup
        therefore increasing their priority.
        """
        keys = torch.tensor(
            [
                [0, 0, 2.1, 0, 0],
                [0, 0, 2.2, 0, 0],
                [0, 0, 1.9, 0, 0],
                [1, 0, 0.0, 0, 0],
                [0, 1, 0.0, 0, 0],
                [0, 0, 0.0, 1, 0],
                [0, 0, 0.0, 0, 1],
            ]
        )
        values = torch.tensor([2, 2, 2, 0, 0, 0, 0]).unsqueeze(1)
        N, key_size = keys.shape
        assert len(keys) == len(values) == N

        # filll DND
        dnd = DND(key_size, torch.device("cpu"), max_size=N, knn_no=3)
        for k, v in zip(keys, values):
            dnd.write(k, v, update_rule=lambda old, new: new)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

        dnd.rebuild_tree()
        dnd.lookup(torch.tensor([[0, 0, 2.0, 0, 0]])).squeeze().item()
        should_be = [0, 1, 2]
        used_keys = [k for k,v in dnd._priority.items() if v == 1]
        for idx in used_keys:
            self.assertIn(idx, should_be)

    def test_prioritized_pop(self):
        """ Insert two clusters and use three cluster elements for a lookup
        therefore increasing their priority. Then pop the unused keys.
        """
        keys = torch.tensor(
            [
                [0, 0, 2.1, 0, 0],
                [0, 0, 2.2, 0, 0],
                [0, 0, 1.9, 0, 0],
                [1, 0, 0.0, 0, 0],
                [0, 1, 0.0, 0, 0],
                [0, 0, 0.0, 1, 0],
                [0, 0, 0.0, 0, 1],
            ]
        )
        values = torch.tensor([2, 2, 2, 0, 0, 0, 0]).unsqueeze(1)
        N, key_size = keys.shape
        assert len(keys) == len(values) == N

        # filll DND
        dnd = DND(key_size, torch.device("cpu"), max_size=N, knn_no=3)
        for k, v in zip(keys, values):
            dnd.write(k, v, update_rule=lambda old, new: new)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

        dnd.rebuild_tree()
        dnd.lookup(torch.tensor([[0, 0, 2.0, 0, 0]])).squeeze().item()

        print(dnd._priority)
        dnd.write(torch.ones(1, 5), 2)
        print(dnd._dict.keys())

    def test_update_rule(self):
        """ Insert N equal elements in a TensorDict of size N and check the
            value is right.
        """
        N, key_size = 5, 12
        key = torch.rand(1, key_size)
        data = {key.clone(): val for val in range(N)}

        dnd = DND(key_size, torch.device("cpu"), max_size=N)
        for k, v in data.items():
            dnd.write(k, v, update_rule=lambda old, new: old + new)

        # its single value should be the sum of all values inserted
        self.assertEqual(dnd._dict[key], reduce(lambda a, b: a + b, range(N)))

    def test_clusters(self):
        """ Insert gaussian clusters and retrieve the proper value associated
        with each.
        """
        # generate clusters
        means, std = [-5.0, 5.0, 10.0], 0.5
        N_ = 100  # N per cluster
        N, key_size = N_ * len(means), 5
        keys = torch.cat([torch.randn(N_, key_size) * std + mu for mu in means])
        values = torch.cat([torch.zeros(N_, 1) + mu for mu in means])
        rnd_idxs = torch.randperm(len(keys))  # shuffle the rows
        keys = keys[rnd_idxs]
        values = values[rnd_idxs]

        # filll DND
        dnd = DND(key_size, torch.device("cpu"), max_size=N, knn_no=3)
        for k, v in zip(keys, values):
            dnd.write(k, v, update_rule=lambda old, new: new)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

        # querry DND
        dnd.rebuild_tree()
        values = [dnd.lookup(torch.ones(1, key_size).fill_(mu)) for mu in means]
        values = [v.squeeze().item() for v in values]

        print(f"clusters: {means}\nvalues:   {values}")

        for value, expected in zip(values, means):
            self.assertAlmostEqual(value, expected, places=self.precision)

    def test_binary_cluster(self):
        """ Insert one-hot vectors and retrieve the proper value associated
        with each.
        """
        keys = torch.tensor(
            [
                [0, 0, 2.1, 0, 0],
                [0, 0, 2.2, 0, 0],
                [0, 0, 1.9, 0, 0],
                [1, 0, 0.0, 0, 0],
                [0, 1, 0.0, 0, 0],
                [0, 0, 0.0, 1, 0],
                [0, 0, 0.0, 0, 1],
            ]
        )
        values = torch.tensor([2, 2, 2, 0, 0, 0, 0]).unsqueeze(1)
        N, key_size = keys.shape
        assert len(keys) == len(values) == N

        # filll DND
        dnd = DND(key_size, torch.device("cpu"), max_size=N, knn_no=3)
        for k, v in zip(keys, values):
            dnd.write(k, v, update_rule=lambda old, new: new)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

        dnd.rebuild_tree()
        value = dnd.lookup(torch.tensor([[0, 0, 2.0, 0, 0]])).squeeze().item()
        self.assertEqual(value, 2)

    def test_backprop(self):
        """ Backprop through the DND. """
        keys = torch.tensor(
            [
                [0, 0, 2.1, 0, 0],
                [0, 0, 2.2, 0, 0],
                [0, 0, 1.9, 0, 0],
                # [1, 0, 0.0, 0, 0],
                # [0, 1, 0.0, 0, 0],
                # [0, 0, 0.0, 1, 0],
                # [0, 0, 0.0, 0, 1],
            ]
        )
        values = torch.tensor([2.1, 2.3, 2.]).unsqueeze(1)
        N, key_size = keys.shape
        assert len(keys) == len(values) == N

        # filll DND
        dnd = DND(key_size, torch.device("cpu"), max_size=N, knn_no=3)
        for k, v in zip(keys, values):
            dnd.write(k, v, update_rule=lambda old, new: new)

        self.assertEqual(len(dnd), N)  # the DND should be of size N

        # high dimensional new state
        obs = torch.tensor([[0, 0, 1.0, 2.0, 1.0, 0, 0]])
        params = torch.randn(key_size, obs.shape[1])
        params.requires_grad_(True)

        h = obs @ params.t()
        dnd.rebuild_tree()
        val = dnd.lookup(h)

        h.register_hook(lambda g: print("h hook:   ", g))
        val.register_hook(lambda g: print("v hook:   ", g))

        # print("h:      ", h)
        # print("h.grad? ", h.requires_grad)
        # print("h.grad: ", h.grad)
        # print("val:    ", val)
        # loss = (val - torch.tensor([0.0])).pow(2).squeeze()
        # print("loss:   ", loss)
        # loss.backward()
        # print("h.grad: ", h.grad)
        # print("w.grad: ", params.grad)
        raise NotImplementedError("Implement DND backprop test!")
