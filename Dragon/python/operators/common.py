# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
from six.moves import range as xrange

from dragon.core.tensor import Tensor, GetTensorName
import dragon.core.workspace as ws


def At(inputs, indices=[], axis=0, acc_gradient=False, **kwargs):

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    if isinstance(inputs, list):
        if len(inputs) != 2:
            raise TypeError('At Operator accpets a list of 2 Tensors')
    elif isinstance(inputs, Tensor):
        if not isinstance(indices, list):
            raise TypeError('At Operator accepts a list of indices')
        indices = np.array(indices, dtype=np.float32)
        tensor = GetTensorName()
        ws.FeedTensor(tensor, indices)
        kwargs['inputs'] = [kwargs['inputs'], Tensor(tensor)]

    output = Tensor.CreateOperator(op_type='At', nout=1, **kwargs)

    if isinstance(inputs, Tensor):
        if inputs.shape is not None:
            output.shape = inputs.shape[:]
            output.shape[axis] = len(indices)

    return output


def Crop(inputs, shape=(), shape_like=None, axis=2, offsets=(), **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('Crop Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    if shape_like is not None:
        if not isinstance(shape_like, Tensor):
            raise TypeError('shape_like can only take a Tensor.')
        kwargs['extra_inputs'] = shape_like
        kwargs['shape_like'] = shape_like.name

    return Tensor.CreateOperator(nout=1, op_type='Crop', **kwargs)


def Slice(inputs, axis=1, num_output=1, **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('Slice Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    outputs = Tensor.CreateOperator(op_type='Slice', nout=num_output, **kwargs)

    if inputs.shape is not None:
        if inputs.shape[axis] is not None:
            for i in xrange(len(outputs)):
                outputs[i].shape = inputs.shape[:]
                outputs[i].shape[axis] /= num_output

    return outputs


def Concat(inputs, axis=1, **kwargs):
    if not isinstance(inputs, list): inputs = [inputs]

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    kwargs['num_input'] = len(inputs)

    output = Tensor.CreateOperator(nout=1, op_type='Concat', **kwargs)
    if all(input.shape is not None for input in inputs):
        if all(input.shape[axis] is not None for input in inputs):
            output.shape = inputs[0].shape[:]
            for i in xrange(1, len(inputs)):
                output.shape[axis] += inputs[i].shape[axis]

    return output


def Reduce(inputs, axis=-1, operation='NONE', keep_dims=False, **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('Reduce Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Reduce', **kwargs)

    if inputs.shape is not None:
        if axis == -1: output.shape = [1]
        else:
            output.shape = inputs.shape[:]
            if keep_dims: output.shape[axis] = 1
            else: del output.shape[axis]

    return output


def Sum(inputs, axis=-1, keep_dims=False, **kwargs):
    return Reduce(inputs, axis, 'SUM', keep_dims, **kwargs)


def Mean(inputs, axis=-1, keep_dims=False, **kwargs):
    return Reduce(inputs, axis, 'MEAN', keep_dims, **kwargs)


def Transpose(inputs, perm=None, **kwargs):
    if not isinstance(inputs, Tensor):
        raise ValueError('Transpose Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    if kwargs['perm'] is None: kwargs['perm'] = [1, 0]

    output = Tensor.CreateOperator(nout=1, op_type='Transpose', **kwargs)

    if inputs.shape is not None:
        if len(inputs.shape) != len(kwargs['perm']):
            raise ValueError('input ndim is {}, but perm provide {}'. \
                             format(len(inputs.shape), len(kwargs['perm'])))
        output.shape = inputs.shape[:]
        for i, axis in enumerate(kwargs['perm']):
            output.shape[i] = inputs.shape[axis]

    return output


def Tile(inputs, multiples, **kwargs):
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Tile Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Tile', **kwargs)

    if inputs.shape is not None:
        if len(inputs.shape) != len(multiples):
            raise ValueError('input ndim is {}, but multiples provide {}'. \
                             format(len(inputs.shape), len(multiples)))
        output.shape = inputs.shape[:]
        for i, multiple in enumerate(multiples):
            output.shape[i] *= multiple

    return output


def Flatten(inputs, axis=0, num_axes=-1, **kwargs):
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Flatten Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Flatten', **kwargs)

    if inputs.shape is not None:
        if num_axes == -1: num_axes = len(inputs.shape) - axis
        elif num_axes == 0:
            raise ValueError('num_axes must > 0 or be -1.')
        num_flatten = np.prod(inputs.shape[axis : axis + num_axes])
        output.shape = inputs.shape[: axis] + [num_flatten] + inputs.shape[axis + num_axes :]

    return output


def Reshape(inputs, shape, **kwargs):
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Reshape Operator accepts a Tensor as inputs')

    if not isinstance(shape, tuple) and not isinstance(shape, list):
        raise TypeError('Reshape dims must be a tuple or list')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Reshape', **kwargs)

    if inputs.shape is not None:
        output.shape = [1] * len(shape)
        for i, s in enumerate(shape):
            if s == -1: output.shape[i] = 1
            else: output.shape[i] = s

    return output


def ExpandDims(inputs, axis=-1, **kwargs):
    if not isinstance(inputs, Tensor):
        raise RuntimeError('ExpandDims Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='ExpandDims', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        output.shape.insert(axis, 1)

    return output


def Shape(inputs, **kwargs):
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Shape Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Shape', **kwargs)

    if inputs.shape is not None:
        output.shape = [len(inputs.shape)]

    return output

