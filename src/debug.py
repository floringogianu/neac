""" Debug utils.
"""
import gc
from collections import defaultdict, OrderedDict
import torch
import numpy as np


def count_tensors(numpy=False):
    stuff = defaultdict(int)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                stuff[f"{type(obj)} {obj.size()}"] += 1
            if numpy:
                if isinstance(obj, np.ndarray):
                    stuff[f"{type(obj)} {obj.shape}"] += 1
        except:
            pass
    stuff = OrderedDict(
        sorted(stuff.items(), key=lambda kv: kv[1], reverse=True)
    )
    print(80 * "-")
    print(f"Found {sum(stuff.values())} items in memory.")
    print(80 * "-")
    for k, v in stuff.items():
        print(f"{k:60} {v} items.")
    print(80 * "-")

