# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
from six.moves import range as xrange
from dragon.core.tensor import GetTensorName
import dragon.core.workspace as ws

from . import *

def At(inputs, indices, axis=0, acc_gradient=False, **kwargs):
    """1D At interface of NDArray.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    indices : list or Tensor
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
    arguments = ParseArguments(locals())

    if isinstance(inputs, list): CheckInputs(inputs, 2)
    elif isinstance(inputs, Tensor):
        if not isinstance(indices, list):
            raise ValueError('The type of indices should be list.')
        indices = np.array(indices, dtype=np.float32)
        tensor = GetTensorName()
        ws.FeedTensor(tensor, indices)
        arguments['inputs'] = [arguments['inputs'], Tensor(tensor)]

    output = Tensor.CreateOperator(op_type='At', nout=1, **arguments)

    if isinstance(inputs, Tensor):
        if inputs.shape is not None:
            output.shape = inputs.shape[:]
            output.shape[axis] = len(indices)

    return output


def RandomPick(inputs, max_samples=1, axis=0, **kwargs):
    """1D RandomPick interface of NDArray.

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


def Crop(inputs, shape, shape_like=None, axis=2, offsets=(), **kwargs):
    """2D Crop interface interface of NDArray.

    Set ``shape`` to None, if you want to use ``shape_like``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    shape : list or None
        The shape of cropping.
    shape_like : Tensor or None
        The shape of cropping. Default is ``None`` (Use ``shape``).
    axis : int
        The start axis of cropping.
    offsets : int or list of int
        The offsets. A single value or list of values.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
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
            for i in xrange(len(outputs)):
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
    arguments['num_input'] = len(inputs)

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
    arguments['num_input'] = len(inputs)

    output = Tensor.CreateOperator(nout=1, op_type='Concat', **arguments)

    if all(input.shape is not None for input in inputs):
        if all(input.shape[axis] is not None for input in inputs):
            output.shape = inputs[0].shape[:]
            for i in xrange(1, int(len(inputs))):
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
        if axis == -1: output.shape = [1]
        else:
            output.shape = inputs.shape[:]
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
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Argmax', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if top_k > 1: output.shape[axis] = top_k
        else: del output.shape[axis]

    return output


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
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Argmin', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if top_k > 1: output.shape[axis] = top_k
        else: del output.shape[axis]

    return output


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
    if perms is None: arguments['perms'] = []

    output = Tensor.CreateOperator(nout=1, op_type='Transpose', **arguments)

    if inputs.shape is not None:
        if perms is None: perms = list(range(((len(inputs.shape)) - 1), -1, -1))
        if len(inputs.shape) != len(perms):
            raise ValueError('The ndim of inputs is {}, but perms provide {}'. \
                             format(len(inputs.shape), len(perms)))
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
    repeats : int
        The magnitude of repeating.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Repeat', **arguments)

    if inputs.shape is not None:
        if axis == -1:
            total_count = np.prod(inputs.shape)
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
    multiples : list of int
        The multiple of each axis.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Tile', **arguments)

    if inputs.shape is not None:
        if len(inputs.shape) != len(multiples):
            raise ValueError('input ndim is {}, but multiples provide {}'. \
                             format(len(inputs.shape), len(multiples)))
        output.shape = inputs.shape[:]
        for i, multiple in enumerate(multiples):
            output.shape[i] *= multiple

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
        if keep_axes is not None:
            if keep_axes > len(inputs.shape):
                raise ValueError('The total number of axes is {}, can not keep {}.'
                                 .format(len(inputs.shape), keep_axes))
            total_count = np.prod(inputs.shape)
            output.shape = []
            for i in xrange(keep_axes - 1):
                output.shape.append(inputs.shape[i])
                total_count *= inputs.shape[i]
            if total_count != 1:
                output.shape.append(np.long(total_count))
        else:
            if num_axes == -1: num_axes = len(inputs.shape) - axis
            elif num_axes == 0:
                raise ValueError('num_axes must > 0 or be -1.')
            num_flatten = np.prod(inputs.shape[axis : axis + num_axes])
            output.shape = inputs.shape[: axis] + [num_flatten] + inputs.shape[axis + num_axes :]

    return output


def Reshape(inputs, shape, **kwargs):
    """Reshape the dimensions of input.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    shape : list or tuple
        The new shape.

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

    if not isinstance(shape, tuple) and not isinstance(shape, list):
        raise TypeError('The type of dims must be a tuple or list.')

    output = Tensor.CreateOperator(nout=1, op_type='Reshape', **arguments)

    if inputs.shape is not None:
        output.shape = [1] * len(shape)
        for i, s in enumerate(shape):
            if s == -1: output.shape[i] = 1
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
        The data type. ``FLOAT32`` or ``INT32``.

    Returns
    -------
    Tensor
        The vector.

    """
    arguments = ParseArguments(locals())
    arguments['extra_inputs'] = []
    if not isinstance(start, Tensor): arguments['static_start'] = int(start)
    else:
        arguments['dynamic_start'] = start.name
        arguments['extra_inputs'].append(start)
    if stop is not None:
        if not isinstance(stop, Tensor): arguments['static_stop'] = int(stop)
        else:
            arguments['dynamic_stop'] = stop.name
            arguments['extra_inputs'].append(stop)
        del arguments['stop']
    if not isinstance(step, Tensor): arguments['static_step'] = int(step)
    else:
        arguments['dynamic_step'] = step.name
        arguments['extra_inputs'].append(step)
    del arguments['start']; del arguments['step']

    output = Tensor.CreateOperator([], nout=1, op_type='Arange', **arguments)

    if 'static_start' in arguments and \
       'static_step' in arguments:
        if 'dynamic_stop' not in arguments:
            if stop is None: stop = start; start = 0
            count = (stop - start - 1) / step + 1
            output.shape = [np.long(count)]

    return output