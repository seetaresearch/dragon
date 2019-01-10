# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
import numpy as np


def constant(value, dtype=None, shape=None, name=None, verify_shape=False):
    if dtype is not None:
        if isinstance(value, np.ndarray):
            feed = value.astype(dtype.as_numpy_dtype)
        elif isinstance(value, list):
            feed = np.array(value, dtype.as_numpy_dtype)
        else:
            feed = np.array([value], dtype.as_numpy_dtype)
    else:
        if isinstance(value, np.ndarray): feed = value
        else:
            feed = np.array(value)
            # Discard the default float64
            if feed.dtype == np.float64:
                feed = feed.astype(np.float32)

    # Determine the shape
    if shape is not None:
        if feed.size == 1:
            # Case 1: Broadcast with scalar value
            c = feed.flatten()[0]
            feed = np.zeros(shape, feed.dtype)
            feed.fill(c)
        else:
            # Case 2: Reshape directly
            if verify_shape:
                if shape is not None:
                    if len(shape) != len(value.shape):
                        raise RuntimeError(
                            'The constant was limited to {} dimensions, \
                                while feed a value with {} dimensions.'.
                                    format(len(shape), len(value.shape)))
                    for i in range(len(shape)):
                        if shape[i] is None: continue
                        if shape[i] != value.shape[i]:
                            raise RuntimeError(
                                'The shape of constant was limited as (' +
                                ','.join([str(dim) for dim in shape]) + '), ' +
                                    'while feed a value with (' + ','.join([str(dim) for dim in value.shape]) + ').')
            feed = feed.reshape(shape)

    defined_name = dragon.workspace.GetDummyName(
        dragon.get_default_name_scope() +
            (name if name else 'Const'),
                suffix=':0', domain='Tensor')

    # Feed into the workspace
    tensor = dragon.Tensor.Ref(
        name=defined_name,
            shape=list(feed.shape),
                dtype=str(feed.dtype))
    tensor.set_value(feed)
    return tensor