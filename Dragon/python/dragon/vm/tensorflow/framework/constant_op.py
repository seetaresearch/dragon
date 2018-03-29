# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

__all__ = ['constant']

import numpy as np

import dragon.core.workspace as ws
from dragon.core.tensor import Tensor

from dragon.vm.tensorflow.framework import dtypes


def constant(value, dtype=None, shape=None, name=None, verify_shape=False):
    # determine the data type
    if dtype == None: dtype = dtypes.float32
    if isinstance(value, np.ndarray):
        feed = value.astype(dtype.as_numpy_dtype)
    elif isinstance(value, list):
        feed = np.array(value, dtype.as_numpy_dtype)
    else:
        feed = np.array([value], dtype.as_numpy_dtype)

    # determine the shape
    if shape is not None:
        if feed.size == 1:
            # case 1: broadcast with scalar value
            c = feed[0]
            feed = np.zeros(shape, dtype.as_numpy_dtype)
            feed.fill(c)
        else:
            # case 2: reshape directly
            if verify_shape:
                if shape is not None:
                    if len(shape) != len(value.shape):
                        raise RuntimeError('The constant was limited to {} dimensions, \
                                                while feed a value with {} dimensions.'.
                                           format(len(shape), len(value.shape)))
                    for i in xrange(len(shape)):
                        if shape[i] is None: continue
                        if shape[i] != value.shape[i]:
                            raise RuntimeError('The shape of constant was limited as (' +
                                               ','.join([str(dim) for dim in shape]) + '), ' +
                                               'while feed a value with (' + ','.join([str(dim) for dim in value.shape]) + ').')
            feed = feed.reshape(shape)

    # feed to VM
    tensor = Tensor(name)
    tensor.shape = list(feed.shape)
    ws.FeedTensor(tensor, feed)
    return tensor
