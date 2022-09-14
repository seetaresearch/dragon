# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/input_spec.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.framework import tensor_shape


class InputSpec(object):
    """Spec to describe properties of a layer input."""

    def __init__(
        self,
        dtype=None,
        shape=None,
        ndim=None,
        max_ndim=None,
        min_ndim=None,
        axes=None,
    ):
        self.dtype = dtypes.as_dtype(dtype).name if dtype is not None else None
        if shape is not None:
            self.ndim = len(shape)
            self.shape = shape
        else:
            self.ndim = ndim
            self.shape = None
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        try:
            axes = axes or {}
            self.axes = {int(k): axes[k] for k in axes}
        except (ValueError, TypeError):
            raise TypeError('The keys in axes must be integers.')
        if self.axes and (self.ndim is not None or self.max_ndim is not None):
            max_dim = (self.ndim if self.ndim else self.max_ndim) - 1
            max_axis = max(self.axes)
            if max_axis > max_dim:
                raise ValueError(
                    'Axis {} is greater than the maximum allowed value: {}'
                    .format(max_axis, max_dim))

    def __repr__(self):
        spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
                ('shape=' + str(self.shape)) if self.shape else '',
                ('ndim=' + str(self.ndim)) if self.ndim else '',
                ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
                ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
                ('axes=' + str(self.axes)) if self.axes else '']
        return 'InputSpec(%s)' % ', '.join(x for x in spec if x)


def assert_input_compatibility(input_spec, inputs, layer_name):
    """Check the input tensors according to the spec."""
    if not input_spec:
        return
    inputs = nest.flatten(inputs)
    input_spec = nest.flatten(input_spec)
    # Check the number of inputs and specs.
    if len(inputs) != len(input_spec):
        raise ValueError(
            'Layer ' + layer_name + ' expects ' +
            str(len(input_spec)) + ' inputs, '
            'but it received ' + str(len(inputs)) + ' input tensors.')
    # For each pair of input and spec.
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
        if spec is None:
            continue
        x_shape = tensor_shape.TensorShape(x.shape)
        # Check the shape is not ``None``.
        if (spec.ndim is not None or
                spec.min_ndim is not None or
                spec.max_ndim is not None):
            if x_shape.ndims is None:
                raise ValueError(
                    'Input ' + str(input_index) + ' of layer ' +
                    layer_name + ' is incompatible with the layer: '
                    'its rank is undefined, but the layer requires a '
                    'defined rank.')
        # Check the total number of dimensions.
        if spec.ndim is not None:
            ndim = x_shape.ndims
            if ndim != spec.ndim:
                raise ValueError(
                    'Input ' + str(input_index) + ' of layer ' +
                    layer_name + ' is incompatible with the layer: '
                    'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                    str(ndim) + '. Full shape received: ' +
                    str(x_shape.as_list()))
        # Check the max number of dimensions.
        if spec.max_ndim is not None:
            ndim = x_shape.ndims
            if ndim is not None and ndim > spec.max_ndim:
                raise ValueError(
                    'Input ' + str(input_index) + ' of layer ' +
                    layer_name + ' is incompatible with the layer: '
                    'expected max_ndim=' + str(spec.max_ndim) +
                    ', found ndim=' + str(ndim))
        # Check the min number of dimensions.
        if spec.min_ndim is not None:
            ndim = x_shape.ndims
            if ndim is not None and ndim < spec.min_ndim:
                raise ValueError(
                    'Input ' + str(input_index) + ' of layer ' +
                    layer_name + ' is incompatible with the layer: '
                    ': expected min_ndim=' + str(spec.min_ndim) +
                    ', found ndim=' + str(ndim) +
                    '. Full shape received: ' +
                    str(x_shape.as_list()))
        # Check the data type.
        if spec.dtype is not None:
            if x.dtype != spec.dtype:
                raise ValueError(
                    'Input ' + str(input_index) + ' of layer ' +
                    layer_name + ' is incompatible with the layer: '
                    'expected dtype=' + str(spec.dtype) +
                    ', found dtype=' + str(x.dtype))
        # Check axes.
        if spec.axes:
            shape = x_shape.as_list()
            if shape is not None:
                for axis, value in spec.axes.items():
                    if hasattr(value, 'value'):
                        value = value.value
                    if value is not None and shape[int(axis)] not in {value, None}:
                        raise ValueError(
                            'Input ' + str(input_index) + ' of layer ' + layer_name + ' is'
                            ' incompatible with the layer: expected axis ' + str(axis) +
                            ' of input shape to have value ' + str(value) +
                            ' but received input with shape ' + str(shape))
        # Check the determined dimensions.
        if spec.shape is not None:
            shape = x_shape.as_list()
            if shape is not None:
                for spec_dim, dim in zip(spec.shape, shape):
                    if spec_dim is not None and dim is not None:
                        if spec_dim != dim:
                            raise ValueError(
                                'Input ' + str(input_index) +
                                ' is incompatible with layer ' + layer_name +
                                ': expected shape=' + str(spec.shape) +
                                ', found shape=' + str(shape))
