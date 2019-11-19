import unittest

import torch
from src.dnd import TensorDict


class TestTensorDict(unittest.TestCase):
    """ TestCase for the IOU function. """

    def __init__(self, *args, **kwargs):
        super(TestTensorDict, self).__init__(*args, **kwargs)
        self.precision = 4

    def test_insert_n_different(self):
        """ Insert N different elements in a TensorDict of size N.
        """
        N, key_size = 5, 12
        data = torch.rand(N, key_size)
        d = TensorDict(N, key_size, torch.device("cpu"))
        for i, key in enumerate(data):
            d[key] = i
        for value, expected in zip(d.values().squeeze().tolist(), range(N)):
            self.assertEqual(value, expected)

    def test_insert_n_equal(self):
        """ Insert N equal elements in a TensorDict of size N.
        """
        N, key_size = 5, 12
        key = torch.rand(1, key_size)
        data = {key.clone(): val for val in range(N)}

        d = TensorDict(N, key_size, torch.device("cpu"))
        for k, v in data.items():
            d[k] = v

        self.assertEqual(len(d), 1)  # should have one element
        self.assertEqual(d[key].item(), N - 1)  # it's value should be N-1

    def test_maxed_out(self):
        """ Test it raises error when at capacity.
        """
        N, key_size = 5, 12
        data = torch.rand(N + 1, key_size)

        d = TensorDict(N, key_size, torch.device("cpu"))
        for i in range(N):
            d[data[i]] = i

        def f():
            d[data[N]] = N

        self.assertRaises(Exception, f)

