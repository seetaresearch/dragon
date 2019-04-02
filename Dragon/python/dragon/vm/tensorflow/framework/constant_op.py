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

import numpy

from dragon.core import scope as _scope
from dragon.core import workspace as _workspace
from dragon.core.tensor import Tensor as _Tensor


def constant(
    value,
    dtype=None,
    shape=None,
    name=None,
    verify_shape=False,
):
    if dtype is not None:
        if isinstance(value, numpy.ndarray):
            value = value.astype(dtype.as_numpy_dtype)
        else:
            value = numpy.array(value, dtype.as_numpy_dtype)
    else:
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value)
            # Discard the default float64
            if value.dtype == numpy.float64:
                value = value.astype(numpy.float32)

    # Determine the shape
    if shape is not None:
        if value.size == 1:
            # Case 1: Broadcast with scalar value
            scalar = value.flatten()[0]
            value = numpy.empty(shape, value.dtype)
            value.fill(scalar)
        else:
            # Case 2: Reshape directly
            if verify_shape:
                if shape is not None:
                    if len(shape) != len(value.shape):
                        raise RuntimeError(
                            'The constant was limited to {} dimensions, ' \
                            'while feed a value with {} dimensions.'
                            .format(len(shape), len(value.shape)))
                    for i in range(len(shape)):
                        if shape[i] is None: continue
                        if shape[i] != value.shape[i]:
                            raise RuntimeError(
                                'The shape of constant was limited as (' +
                                ','.join([str(dim) for dim in shape]) + '), ' +
                                'while feed a value with (' +
                                ','.join([str(dim) for dim in value.shape]) + ').')
            value = value.reshape(shape)

    # Get a available name
    defined_name = \
        _workspace.GetDummyName(
            basename=_scope.get_default_name_scope() +
                (name if name else 'Const'),
            suffix=':0', domain='Tensor')

    # Feed into the workspace
    return _Tensor.Ref(
        name=defined_name,
            shape=list(value.shape),
                dtype=str(value.dtype)
    ).set_value(value)