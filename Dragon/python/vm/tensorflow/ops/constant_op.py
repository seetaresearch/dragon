# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = ['constant']

import numpy as np

import dragon.core.workspace as ws
from dragon.core.tensor import Tensor
from ..core import dtypes


def constant(value, dtype=None, shape=None, name=None):
    if dtype == None: dtype = dtypes.float32
    if isinstance(value, np.ndarray): feed = value.astype(dtype)
    elif isinstance(value, list): feed = np.array(value, dtype)
    else: feed = np.array([value], dtype)
    if shape is not None:
      if feed.size == 1:
        c = feed[0]
        feed = np.zeros(shape, dtype)
        feed.fill(c)
      else: feed = feed.reshape(shape)
    tensor = Tensor(name)
    tensor.shape = list(feed.shape)
    ws.FeedTensor(tensor, feed)
    return tensor
