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

import numpy as np

from . import *


def Gather(inputs, indices, axis=0, acc_gradient=False, **kwargs):
    """Gather the input according to the indices along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    indices : int, list or Tensor
        The indices to form output tensor.
    axis : int
        The start axis.
    acc_gradient : boolean
        Whether to accumulate gradients.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    arguments['inputs'] = [arguments['inputs'],
                           Tensor.Convert(indices, dtype='int32')]
    arguments['indices'] = None

    output = Tensor.CreateOperator(op_type='Gather', nout=1, **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if not isinstance(indices, Tensor):
            if not isinstance(indices, (list, tuple)):
                indices = [indices]
            output.shape[axis] = len(indices)
        else:
            output.shape[axis] = None

    return output


def RandomPick(inputs, max_samples=1, axis=0, **kwargs):
    """Randomly pick the input along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    max_samples : int
        The max samples to pick.
    axis : int
        The start axis.

    Returns
    -------
    Tensor
        The output tensor, sampled randomly.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    outputs = Tensor.CreateOperator(op_type='RandomPick', nout=2, **arguments)

    if inputs.shape is not None:
        outputs[0].shape = inputs.shape[:]
        outputs[0].shape[axis] = max_samples
        outputs[1].shape = [max_samples]

    return outputs


def Crop(inputs, starts, ends, start_axis=None,
         offsets=None, shape=None, shape_like=None, **kwargs):
    """Crop the input according to the given starts and ends.

    Set ``starts`` and ``ends`` to None, if want to use ``start_axis``, ``offsets`` and ``shape``.

    Set ``shape`` to None, if you want to use ``shape_like``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    starts : int/Tensor, list of int/Tensor, or None
        The starts.
    starts : int/Tensor, list of int/Tensor, or None
        The ends.
    start_axis : int or None
        The axis to start. Default is ``None`` (Disabled).
    offsets : int, list of int or None
        The offsets. Ignore the axes before ``start_axis``.
    shape : list, tuple or None
        The referring shape. Use ``-1`` to represent the unknown dimensions.
    shape_like : Tensor or None
       The referring shape. Default is ``None`` (Disabled).

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> x = Tensor('x', dtype='float32').Variable()
    >>> x.set_value(np.arange(1, 25).reshape((1, 2, 3, 4)))
    >>> y = Crop(x, starts=[0, 1, 0, 2], ends=[1, 2, 0, 0])
    >>> y = x[0:1, 1:2, :, 2:] # the same as above
    >>> y = Crop(x, None, None, start_axis=1, offsets=(1, 0, 2), shape=(-1, 1, 3, 2)) # the same as above

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    if starts is not None:
        AddArgumentsWithDesc(arguments, starts, 'starts', 'int32', as_target=True)
    if ends is not None:
        AddArgumentsWithDesc(arguments, ends, 'ends', 'int32', as_target=True)
    if offsets is not None:
        if not isinstance(offsets, (list, tuple)):
            arguments['offsets'] = [offsets]
    if shape is None: arguments['shape'] = []
    if shape_like is not None:
        if not isinstance(shape_like, Tensor):
            raise ValueError('The type of shape_like should be Tensor.')
        arguments['extra_inputs'] = shape_like
        arguments['shape_like'] = shape_like.name

    return Tensor.CreateOperator(nout=1, op_type='Crop', **arguments)


def Slice(inputs, axis=1, num_output=1, **kwargs):
    """Slice interface of NDArray.

    The dimension of specific axis should be divided by ``num_output``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to slice.
    num_output : int
        The number of slices.

    Returns
    -------
    Tensor or list of Tensor
        The outputs.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    outputs = Tensor.CreateOperator(op_type='Slice', nout=num_output, **arguments)

    if inputs.shape is not None:
        if inputs.shape[axis] is not None:
            for i in range(len(outputs)):
                outputs[i].shape = inputs.shape[:]
                outputs[i].shape[axis] /= num_output

    return outputs


def Stack(inputs, axis=0, **kwargs):
    """Stack the inputs along the given axis.

    All dimensions of inputs should be same.

    The ``axis`` can be negative.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs.
    axis : int
        The axis to stack.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1, INT_MAX)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Stack', **arguments)

    if all(input.shape is not None for input in inputs):
        while axis < 0: axis += (len(inputs[0].shape) + 1)
        output.shape = inputs[0].shape
        output.shape.insert(axis, np.long(len(inputs)))

    return output


def Concat(inputs, axis=1, **kwargs):
    """Concatenate the inputs along the given axis.

    All dimensions except specific ``axis`` should be same.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs.
    axis : int
        The axis to concatenate.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1, INT_MAX)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Concat', **arguments)

    if all(input.shape is not None for input in inputs):
        if all(input.shape[axis] is not None for input in inputs):
            output.shape = inputs[0].shape[:]
            for i in range(1, int(len(inputs))):
                output.shape[axis] += inputs[i].shape[axis]

    return output


def Reduce(inputs, axis=-1, operation='NONE', keep_dims=False, **kwargs):
    """Reduce interface of NDArray.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axis : int
        The axis to reduce. Default is ``-1`` (Compute along all axes).
    operation : str
        The operation, ``SUM`` or ``MEAN``. Default is ``NONE`` (Unknown).
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Reduce', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if axis == -1:
            if keep_dims:
                for i in range(len(output.shape)):
                    output.shape[i] = 1
            else: output.shape = [1]
        else:
            if keep_dims: output.shape[axis] = 1
            else: del output.shape[axis]

    return output


def Sum(inputs, axis=-1, keep_dims=False, **kwargs):
    """Compute the sum along the given axis.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axis : int
        The axis to compute. Default is ``-1`` (Along all axes).
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The sum result.

    See Also
    --------
    `ops.Reduce(*args, **kwargs)`_ - The General Reduce Operator.

    """
    return Reduce(inputs, axis, 'SUM', keep_dims, **kwargs)


def Mean(inputs, axis=-1, keep_dims=False, **kwargs):
    """Compute the mean along the given axis.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axis : int
        The axis to compute. Default is ``-1`` (Along all axes).
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The mean result.

    See Also
    --------
    `ops.Reduce(*args, **kwargs)`_ - The general reduce operator.

    """
    return Reduce(inputs, axis, 'MEAN', keep_dims, **kwargs)


def _ArgReduce(inputs, axis=-1, operation='NONE', top_k=1, keep_dims=False, **kwargs):
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    n_out = 1
    if 'ARG' not in operation:
        n_out = 2; arguments['operation'] = 'ARG' + operation

    outputs = Tensor.CreateOperator(nout=n_out, op_type='ArgReduce', **arguments)
    if 'ARG' not in operation: output = outputs[1]
    else: output = outputs

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if top_k > 1: output.shape[axis] = top_k
        else: del output.shape[axis]

    return output


def Argmax(inputs, axis=-1, top_k=1, keep_dims=False, **kwargs):
    """Compute the indices of maximum elements along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to compute. Default is ``-1`` (Along all axes).
    top_k : int
        The top k results to keep.
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The indices.

    """
    return _ArgReduce(inputs, axis, 'ARGMAX', top_k, keep_dims, **kwargs)


def Max(inputs, axis=-1, top_k=1, keep_dims=False, **kwargs):
    """Compute the values of maximum elements along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to compute. Default is ``-1`` (Along all axes).
    top_k : int
        The top k results to keep.
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The values.

    """
    return _ArgReduce(inputs, axis, 'MAX', top_k, keep_dims, **kwargs)


def Argmin(inputs, axis=-1, top_k=1, keep_dims=False, **kwargs):
    """Compute the indices of minimum elements along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to compute. Default is ``-1`` (Along all axes).
    top_k : int
        The top k results to keep.
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The indices.

    """
    return _ArgReduce(inputs, axis, 'ARGMIN', top_k, keep_dims, **kwargs)


def Min(inputs, axis=-1, top_k=1, keep_dims=False, **kwargs):
    """Compute the values of minimum elements along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to compute. Default is ``-1`` (Along all axes).
    top_k : int
        The top k results to keep.
    keep_dims : boolean
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The values.

    """
    return _ArgReduce(inputs, axis, 'MIN', top_k, keep_dims, **kwargs)


def Transpose(inputs, perms=None, **kwargs):
    """Transpose the input according to the given permutations.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    perms : tuple, list or None
        The permutation. Default is ``None`` (Reverse Dimensions).

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    if perms is None:
        arguments['perms'] = []
    else:
        AddArgumentsWithDesc(arguments, perms, 'perms', 'int32', as_target=True)

    output = Tensor.CreateOperator(nout=1, op_type='Transpose', **arguments)

    if inputs.shape is not None:
        if perms is None: perms = list(range(((len(inputs.shape)) - 1), -1, -1))
        else:
            possible_to_infer_shape = True
            for perm in perms:
                if isinstance(perm, Tensor):
                    possible_to_infer_shape = False
            if possible_to_infer_shape:
                if len(inputs.shape) != len(perms):
                    raise ValueError('The ndim of inputs is {}, but perms provide {}'
                            .format(len(inputs.shape), len(perms)))
                output.shape = inputs.shape[:]
                for i, axis in enumerate(perms):
                    output.shape[i] = inputs.shape[axis]

    return output


def Repeat(inputs, axis=-1, repeats=1, **kwargs):
    """Repeat the input along the given axis.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to repeat. Defaults is ``-1`` (Repeat as Scalar).
    repeats : int or Tensor
        The magnitude of repeating.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    arguments = AddArgumentWithDesc(arguments, repeats, 'repeats', as_target=True)

    output = Tensor.CreateOperator(nout=1, op_type='Repeat', **arguments)

    if inputs.shape is not None and \
            not isinstance(repeats, Tensor):
        if axis == -1:
            fake_shape = inputs.shape[:]
            fake_shape = [1 if dim is None else dim for dim in fake_shape]
            total_count = np.prod(fake_shape)
            output.shape = [total_count * repeats]
        else:
            output.shape = inputs.shape[:]
            output.shape[axis] *= repeats

    return output


def Tile(inputs, multiples, **kwargs):
    """Tile the input according to the given multiples.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    multiples : list
        The multiple of each axis.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    arguments = AddArgumentsWithDesc(arguments, multiples, 'multiples', 'int32', as_target=True)

    output = Tensor.CreateOperator(nout=1, op_type='Tile', **arguments)

    if inputs.shape is not None:
        if len(inputs.shape) != len(multiples):
            raise ValueError('The num of dimensions of input is {}, but provided {}.'
                             .format(len(inputs.shape), len(multiples)))
        output.shape = inputs.shape[:]
        for i, multiple in enumerate(multiples):
            if output.shape[i] is None or \
                isinstance(output.shape[i], Tensor):
                    output.shape[i] = None
            else:
                    output.shape[i] *= multiple

    return output


def Pad(inputs, paddings, mode='CONSTANT', value=0, **kwargs):
    """Pad the input according to the given paddings.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    paddings : list or tuple
        The paddings, 1D/2D list or tuple.
    mode : str
        The padding mode, ``CONSTANT``, ``REFLECT`` or ``EDGE``.
    value : basic numerical type
        The value to use on the ``CONSTANT`` mode.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    pad_l = []; pad_r = []
    for padding in paddings:
        if isinstance(padding, (list, tuple)):
            if len(padding) != 2:
                raise ValueError('The padding should be a list or tuple of length 2.')
            pad_l.append(int(padding[0]))
            pad_r.append(int(padding[1]))
        else:
            pad_l.append(int(padding))
            pad_r.append(int(padding))
    arguments['paddings'] = None
    arguments['pad_l'] = pad_l
    arguments['pad_r'] = pad_r
    arguments['value'] = float(arguments['value'])

    output = Tensor.CreateOperator(nout=1, op_type='Pad', **arguments)

    return output


def OneHot(inputs, depth, on_value=1, off_value=0, **kwargs):
    """Generate the one-hot representation of inputs.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    depth : int
        The depth of one-hot representation.
    on_value : int
        The value when ``indices[j] = i``.
    off_value : int
        The value when ``indices[j] != i``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='OneHot', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        output.shape.append(np.long(depth))

    return output


def Flatten(inputs, axis=0, num_axes=-1, keep_axes=None, **kwargs):
    """Flatten the input along the given axes.

    Set ``keep_axes`` to flatten if shape is dynamic.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The start axis to flatten.
    num_axes : int
        The number of axes to flatten. Default is ``-1`` (Along all axes).
    keep_axes : int or None
        The number of axes to keep. Default is ``None`` (Disabled).

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> a = Tensor(shape=[1, 2, 3, 4]).Variable()
    >>> print Flatten(a, axis=1, num_axes=-1).shape
    >>> [1, 24]

    >>> print Flatten(a, axis=1, num_axes=2).shape
    >>> [1, 6, 4]

    >>> print Flatten(a, keep_axes=1)
    >>> [24]

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Flatten', **arguments)

    if inputs.shape is not None:
        fake_shape = inputs.shape[:]
        fake_shape = [1 if dim is None else dim for dim in fake_shape]
        if keep_axes is not None:
            if keep_axes > len(inputs.shape):
                raise ValueError('The total number of axes is {}, can not keep {}.'
                                 .format(len(inputs.shape), keep_axes))
            total_count = np.prod(fake_shape)
            output.shape = []
            for i in range(keep_axes - 1):
                output.shape.append(inputs.shape[i])
                total_count *= fake_shape[i]
            if total_count != 1:
                output.shape.append(total_count)
        else:
            if num_axes == -1: num_axes = len(inputs.shape) - axis
            elif num_axes == 0:
                raise ValueError('num_axes must > 0 or be -1.')
            num_flatten = np.prod(fake_shape[axis : axis + num_axes])
            output.shape = inputs.shape[: axis] + [num_flatten] + inputs.shape[axis + num_axes :]

    return output


def Reshape(inputs, shape, shape_like=None, **kwargs):
    """Reshape the dimensions of input.

    ``shape`` could be a list of numbers or Tensors.

    Set ``shape`` to ``None``, if you want to use ``shape_like``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    shape : list, tuple or None
        The new shape.
    shape_like: Tensor, str or None
        The tensor for indicating the output shape.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> a = Tensor(shape=[1, 2, 3, 4]).Variable()
    >>> print Reshape(a, shape=[6, 4])
    >>> [6, 4]

    >>> b = Reshape(a, shape=[-1, 4]) # shape will be [6, 4] in the backend
    >>> print b.shape
    >>> [1, 4] # fake dimension at axis 0

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    if shape is not None:
        AddArgumentsWithDesc(arguments, shape, 'shape', 'int32', as_target=True)
    elif shape_like is not None:
        if not isinstance(shape_like, (Tensor, str)):
            raise TypeError('The shape_like should be a Tensor or a name.')
        arguments['shape_like'] = shape_like \
            if isinstance(shape_like, str) else shape_like.name

    output = Tensor.CreateOperator(nout=1, op_type='Reshape', **arguments)

    if inputs.shape is not None:
        possible_to_infer_shape = True
        if shape is not None:
            for dim in shape:
                if isinstance(dim, Tensor):
                    possible_to_infer_shape = False
        if shape_like is not None:
            possible_to_infer_shape = False
        if possible_to_infer_shape:
            output.shape = [1] * len(shape)
            for i, s in enumerate(shape):
                if s == -1: output.shape[i] = 1
                elif s == 0: output.shape[i] = inputs.shape[i]
                else: output.shape[i] = s

    return output


def ExpandDims(inputs, axis=-1, **kwargs):
    """ExpandDims interface of NDArray.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The insert position of new dimension. Default is ``-1`` (Push Back).

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> a = Tensor(shape=[1, 2, 3, 4]).Variable()
    >>> print ExpandDims(a).shape

    >>> print ExpandDims(a, axis=2).shape

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='ExpandDims', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if axis == -1 or axis >= len(inputs.shape):
            output.shape.append(np.long(1))
        else: output.shape.insert(axis, np.long(1))

    return output


def Shape(inputs, **kwargs):
    """Get the dynamic shape of a Tensor.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The dynamic shape.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Shape', **arguments)

    if inputs.shape is not None:
        output.shape = [len(inputs.shape)]

    return output


def Arange(start, stop=None, step=1, dtype='FLOAT32', **kwargs):
    """Return a vector of elements by arange.

    If ``stop`` is None, use the range: [0, start).

    Parameters
    ----------
    start : int or Tensor
        The start of the range.
    stop : int or Tensor
        The stop of range.
    step : int or Tensor
        The interval between two elements.
    dtype : str
        The data type. ``float32`` or ``int32``.

    Returns
    -------
    Tensor
        The vector.

    """
    arguments = ParseArguments(locals())
    arguments['dtype'] = arguments['dtype'].upper()
    arguments = AddArgumentWithDesc(arguments, start, 'start', as_target=True)
    arguments = AddArgumentWithDesc(arguments, step, 'step', as_target=True)
    if stop is not None:
        arguments = AddArgumentWithDesc(arguments, stop, 'stop', as_target=True)

    output = Tensor.CreateOperator([], nout=1, op_type='Arange', **arguments)

    if not isinstance(start, Tensor) and \
        not isinstance(step, Tensor):
        if stop is not None:
            if isinstance(stop, Tensor):
                return output
        else:
            stop = start
            start = 0
        count = int((stop - start - 1) / step) + 1
        output.shape = [count]

    return output