# --------------------------------------------------------
# Theano for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import dragon.core.workspace as ws
from dragon.core.tensor import Tensor, GetTensorName

def shared(value, name=None, borrow=False):
    if name is None: name = GetTensorName()
    if not isinstance(value, np.ndarray):
        raise TypeError('shared variables be a numpy array')
    tensor = Tensor(name).Variable()
    ws.FeedTensor(tensor, value)
    return tensor

