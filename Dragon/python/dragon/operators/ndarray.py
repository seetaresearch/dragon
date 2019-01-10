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

from . import *


@OpSchema.Inputs(1)
def Gather(inputs, indices, axis=0, acc_gradient=False, **kwargs):
    """Gather the input according to the indices along the given axis.

    **Type Constraints**: (*int32*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    indices : int, sequence of (number, Tensor)
        The indices to form output tensor.
    axis : int, optional
        The start axis, can be negative.
    acc_gradient : bool, optional
        Whether to accumulate the gradients.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    arguments['inputs'], arguments['indices'] = [arguments['inputs'],
        Tensor.Convert(indices, dtype='int32')], None

    output = Tensor.CreateOperator('Gather', **arguments)

    try:
        output.shape = inputs.shape[:]
        if not isinstance(indices, Tensor):
            if not isinstance(indices, (list, tuple)):
                indices = [indices]
            output.shape[axis] = len(indices)
        else:
            output.shape[axis] = None
    except:
        pass

    return output


@OpSchema.Inputs(1)
@ArgumentHelper.RepeatedDesc('starts')
@ArgumentHelper.RepeatedDesc('sizes')
def Crop(inputs, starts, sizes, start_axis=None, offsets=None, shape_like=None, **kwargs):
    """Crop the input according to the given starts and sizes.

    Set ``starts`` and ``sizes`` to *None*, if using ``start_axis``, ``offsets`` and ``shape_like``.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    starts : int, Tensor, sequence of (int, Tensor)
        The starts.
    sizes : int, Tensor, sequence of (int, Tensor)
        The crop sizes.
    start_axis : int, optional
        The axis to start.
    offsets : int, sequence of, optional
        The offsets. Ignore the axes before ``start_axis``.
    shape_like : Tensor, optional
       The referring shape.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> import numpy as np
    >>> x = Tensor('x', dtype='float32').Variable()
    >>> x.set_value(np.arange(1, 25).reshape((1, 2, 3, 4)))
    >>> y = Crop(x, starts=[0, 1, 0, 2], sizes=[1, 1, 3, 2])
    >>> a = x[0:1, 1:2, :, 2:] # the same as above

    """
    arguments = ParseArgs(locals())

    if offsets is not None:
        if not isinstance(offsets, (list, tuple)):
            arguments['offsets'] = [offsets]
    if shape_like is not None:
        if not isinstance(shape_like, Tensor):
            raise ValueError('The type of shape_like should be Tensor.')
        arguments['extra_inputs'] = shape_like
        arguments['shape_like'] = shape_like.name

    return Tensor.CreateOperator('Crop', **arguments)


@OpSchema.Inputs(1)
def Slice(inputs, axis=0, num_outputs=1, slice_points=None, **kwargs):
    """Slice the inputs into several parts along the given axis.

    All dimensions except the specified ``axis`` should be same.

    The number of ``slice_points`` should be *len(X.shape) - 1*.

    if ``slice_points`` is *None*, dimension of axis should be divided by ``num_outputs``.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The axis to slice, can be negative.
    num_outputs : int, optional
        The optional number number of slices.
    slice_points : sequence of int, optional
        The optional slice points.

    Returns
    -------
    sequence of Tensor
        The outputs.

    """
    if slice_points is not None and len(slice_points) > 0:
        num_outputs = len(slice_points) + 1
    return Tensor.CreateOperator('Slice', **ParseArgs(locals()))


@OpSchema.Inputs(1, INT_MAX)
def Stack(inputs, axis=0, **kwargs):
    """Stack the inputs along the given axis.

    All the dimensions of inputs should be same.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs.
    axis : int
        The axis to stack, can be negative.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Stack', **ParseArgs(locals()))


@OpSchema.Inputs(1, INT_MAX)
def Concat(inputs, axis=0, **kwargs):
    """Concatenate the inputs along the given axis.

    All the dimensions except the specified ``axis`` should be same.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs.
    axis : int
        The axis to concatenate, can be negative.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Concat', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Reduce(inputs, axes=None, operation='SUM', keep_dims=False, **kwargs):
    """Reduce the inputs along the axis in given axes.

    If ``axes`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axes : int or sequence of int, optional
        The axes to reduce.
    operation : {'SUM', 'MEAN'}, optional
        The operation.
    keep_dims : bool, optional
        Whether to keep dims after reducing.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())
    if axes and not isinstance(axes, (tuple, list)): arguments['axes'] = [axes]
    return Tensor.CreateOperator('Reduce', **arguments)


@OpSchema.Inputs(1)
def Sum(inputs, axes=None, keep_dims=False, **kwargs):
    """Compute the sum along the axis in given axes.

    If ``axes`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axes : int or sequence of int, optional
        The axes to reduce.
    keep_dims : bool, optional
        Whether to keep dims after reducing.

    Returns
    -------
    Tensor
        The sum result.

    See Also
    --------
    `ops.Reduce(*args, **kwargs)`_ - The General Reduce Operator.

    """
    return Reduce(inputs, axes, 'SUM', keep_dims, **kwargs)


@OpSchema.Inputs(1)
def Mean(inputs, axes=None, keep_dims=False, **kwargs):
    """Compute the mean along the axis in given axes.

    If ``axes`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    axes : int or sequence of int, optional
        The axes to reduce.
    keep_dims : bool, optional
        Whether to keep dims after reducing.

    Returns
    -------
    Tensor
        The mean result.

    See Also
    --------
    `ops.Reduce(*args, **kwargs)`_ - The general reduce operator.

    """
    return Reduce(inputs, axes, 'MEAN', keep_dims, **kwargs)


@OpSchema.Inputs(1)
def _ArgReduce(inputs, axis=None, operation='ARGMAX', top_k=1, keep_dims=False, **kwargs):
    arguments = ParseArgs(locals())
    arguments['axis'] = arguments['axis'] if arguments else INT_MAX
    return Tensor.CreateOperator('ArgReduce', num_outputs=2, **arguments)


@OpSchema.Inputs(1)
def ArgMax(inputs, axis=None, top_k=1, keep_dims=False, **kwargs):
    """Compute the indices of maximum elements along the given axis.

    If ``axis`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The axis to compute, can be negative.
    top_k : int, optional
        The top k results to keep.
    keep_dims : bool, optional
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The indices.

    """
    return _ArgReduce(inputs, axis, 'ARGMAX', top_k, keep_dims, **kwargs)[0]


@OpSchema.Inputs(1)
def Max(inputs, axis=None, top_k=1, keep_dims=False, **kwargs):
    """Compute the values of maximum elements along the given axis.

    If ``axis`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The axis to compute, can be negative.
    top_k : int, optional
        The top k results to keep.
    keep_dims : bool, optional
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The values.

    """
    return _ArgReduce(inputs, axis, 'ARGMAX', top_k, keep_dims, **kwargs)[1]


@OpSchema.Inputs(1)
def ArgMin(inputs, axis=None, top_k=1, keep_dims=False, **kwargs):
    """Compute the indices of minimum elements along the given axis.

    If ``axis`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The axis to compute, can be negative.
    top_k : int, optional
        The top k results to keep.
    keep_dims : bool, optional
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The indices.

    """
    return _ArgReduce(inputs, axis, 'ARGMIN', top_k, keep_dims, **kwargs)[0]


@OpSchema.Inputs(1)
def Min(inputs, axis=None, top_k=1, keep_dims=False, **kwargs):
    """Compute the values of minimum elements along the given axis.

    If ``axis`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The axis to compute, can be negative.
    top_k : int, optional
        The top k results to keep.
    keep_dims : bool, optional
        Whether to keep dims after computing.

    Returns
    -------
    Tensor
        The values.

    """
    return _ArgReduce(inputs, axis, 'ARGMIN', top_k, keep_dims, **kwargs)[1]


@OpSchema.Inputs(1)
@ArgumentHelper.RepeatedDesc('perm')
def Transpose(inputs, perm=None, **kwargs):
    """Transpose the input according to the given permutations.

    If ``perm`` is *None*, all the dimensions are reversed.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    perm : sequence of (int, Tensor), optional
        The permutation.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Transpose', **arguments)


@OpSchema.Inputs(1)
@ArgumentHelper.Desc('repeats')
def Repeat(inputs, axis=None, repeats=1, **kwargs):
    """Repeat the input along the given axis.

    If ``axis`` is *None*, flattened results will be returned.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The axis to repeat.
    repeats : int or Tensor, optional
        The magnitude of repeating.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())
    arguments['axis'] = arguments['axis'] if arguments else INT_MAX
    return Tensor.CreateOperator('Repeat', **arguments)


@OpSchema.Inputs(1)
@ArgumentHelper.RepeatedDesc(name='multiples')
def Tile(inputs, multiples, **kwargs):
    """Tile the input according to the given multiples.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    multiples : sequence of (int, Tensor)
        The multiple of each axis.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Tile', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Pad(inputs, pads, mode='CONSTANT', value=0, **kwargs):
    """Pad the input according to the given pads.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    input : Tensor
        The input tensor.
    pads : sequence of number
        The pads, list or tuple.
    mode : {'CONSTANT', 'REFLECT', 'EDGE'}, optional
        The padding mode.
    value : number, optional
        The value that used in the *CONSTANT* mode.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    pads_l, pads_r = [], []
    for pad in pads:
        if isinstance(pad, (list, tuple)):
            if len(pad) != 2:
                raise ValueError(
                    'The pad should be a list or tuple of length 2.')
            pads_l.append(int(pad[0]))
            pads_r.append(int(pad[1]))
        else:
            pads_l.append(int(pad))
            pads_r.append(int(pad))

    arguments['pad_l'], arguments['pad_r'], \
        arguments['pads'], arguments['value'] = \
            pads_l, pads_r, None, float(arguments['value'])

    return Tensor.CreateOperator('Pad', **arguments)


@OpSchema.Inputs(1)
def OneHot(inputs, depth, on_value=1, off_value=0, **kwargs):
    """Generate the one-hot representation of inputs.

    **Type Constraints**: (*int32*, *int64*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    depth : int
        The depth of one-hot representation.
    on_value : int, optional
        The value when ``indices[j] = i``.
    off_value : int, optional
        The value when ``indices[j] != i``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('OneHot', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Flatten(inputs, axis=0, num_axes=-1, keep_axes=None, **kwargs):
    """Flatten the input along the given axes.

    Set ``keep_axes`` to flatten if shape is dynamic.

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The start axis to flatten, can be negative.
    num_axes : int, optional
        The number of axes to flatten. Default is ``-1`` (Along all axes).
    keep_axes : int, optional
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
    arguments = ParseArgs(locals())

    output = Tensor.CreateOperator(op_type='Flatten', **arguments)

    if inputs.shape is not None:
        fake_shape = inputs.shape[:]
        fake_shape = [1 if dim is None else dim for dim in fake_shape]
        if keep_axes is not None:
            if keep_axes > len(inputs.shape):
                raise ValueError(
                    'The total number of axes is {}, can not keep {}.'
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


@OpSchema.Inputs(1)
@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def Reshape(inputs, shape, shape_like=None, **kwargs):
    """Reshape the dimensions of input.

    Set ``shape`` to *None*, if you want to use ``shape_like``.

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    shape : sequence of (int, Tensor)
        The new shape.
    shape_like: str or Tensor, optional
        The tensor for indicating the output shape.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> a = Tensor(shape=[1, 2, 3, 4]).Variable()
    >>> print(Reshape(a, shape=[6, 4]))
    >>> [6, 4]

    >>> b = Reshape(a, shape=[-1, 4]) # shape will be [6, 4] in the backend
    >>> print(b.shape)
    >>> [1, 4] # fake dimension at axis 0

    """
    arguments = ParseArgs(locals())
    if shape_like is not None:
        if not isinstance(shape_like, (Tensor, str)):
            raise TypeError('The shape_like should be a Tensor or a name.')
        arguments['shape_like'] = shape_like \
            if isinstance(shape_like, str) else shape_like.name
    return Tensor.CreateOperator('Reshape', **arguments)


@OpSchema.Inputs(1)
def Squeeze(inputs, axis=None, **kwargs):
    """Remove the dimensions with size 1.

    Set ``axis`` to remove the specific position.

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The specific axis to remove, can be negative.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> a = Tensor(shape=[2, 1, 3, 4]).Variable()
    >>> print(Squeeze(a).shape)
    >>> print(Squeeze(a, axis=0).shape)

    """
    arguments = ParseArgs(locals())

    output = Tensor.CreateOperator(op_type='Squeeze', **arguments)

    if inputs.shape is not None:
        output_shape = []
        if axis: axis += (0 if axis >= 0 else len(inputs.shape))
        for idx, dim in enumerate(inputs.shape[:]):
            if dim != 1 or \
                (axis and dim == 1 and idx != axis):
                    output_shape.append(dim)
        output.shape = output_shape

    return output


@OpSchema.Inputs(1)
def ExpandDims(inputs, axis=0, **kwargs):
    """Expand the new dimension with size 1 to specific axis.

    Negative ``axis`` is equal to *axis = axis + num_axes + 1*.

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int, optional
        The insert axis of new dimension, can be negative.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> a = Tensor(shape=[1, 2, 3, 4]).Variable()
    >>> print(ExpandDims(a).shape)
    >>> print(ExpandDims(a, axis=2).shape)

    """
    return Tensor.CreateOperator('ExpandDims', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Shape(inputs, **kwargs):
    """Get the dynamic shape of a Tensor.

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The dynamic shape.

    """
    return Tensor.CreateOperator('Shape', **ParseArgs(locals()))


@OpSchema.Inputs(0)
@ArgumentHelper.Desc('start')
@ArgumentHelper.Desc('stop')
@ArgumentHelper.Desc('step')
def Arange(start, stop=None, step=1, dtype='float32', **kwargs):
    """Return evenly spaced values within a given interval.

    If ``stop`` is None, use the range: [0, start).

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    start : int or Tensor
        The start of the range.
    stop : int or Tensor, optional
        The stop of range.
    step : int or Tensor, optional
        The interval between two elements.
    dtype : str
        The data type, optional

    Returns
    -------
    Tensor
        A vector with evenly spaced elements.

    """
    arguments = ParseArgs(locals())
    arguments['dtype'] = arguments['dtype'].lower()
    return Tensor.CreateOperator('Arange', [], **arguments)