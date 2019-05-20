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

from dragon.core import mpi
from dragon.vm.torch.tensor import Tensor, _LeafTensor, _Device
from dragon.vm.torch.ops.primitive import MakeDevice, WrapScalar
from dragon.vm.torch.ops.factory import get_module

from dragon.vm.torch.ops.modules.control_flow import (
    Assign, MaskedAssign, Compare
)

from dragon.vm.torch.ops.modules.arithmetic import (
    Fundamental, Accumulate,
    Log, Exp, Sqrt,
    MM, FullyConnected,
    Maximum, Minimum, Clamp,
)

from dragon.vm.torch.ops.modules.init import (
    Fill, RandomUniform, RandomNormal,
)

from dragon.vm.torch.ops.modules.array import (
    Reshape, Squeeze, UnSqueeze, Permute,
    Indexing, Repeat, Concat, Stack,
    IndexSelect, MaskedSelect,
    Reduce, ArgReduce,
    NonZero, Where,
    OneHot, Multinomial,
)

from dragon.vm.torch.ops.modules.update import (
    Accumulate as _Accumulate, Collective, Update,
)

from dragon.vm.torch.ops.modules.vision import (
    Resize2d, RoIPool, RoIAlign,
)


__all__ = [
    'add', 'sub', 'mul', 'div', 'accumulate',
    'maximum', 'minimum', 'clamp',
    'log', 'exp', 'sqrt',
    'mm', 'xw_plus_b',
    'squeeze', 'unsqueeze',
    'mean', 'sum', 'min', 'max', 'topk',
    'nonzero', 'where', 'argmin', 'argmax',
    'gt', 'lt', 'eq', 'ne', 'ge', 'le',
    'cat', 'stack', 'narrow',
    'index_select', 'masked_select',
    'one_hot', 'multinomial',
    'rand', 'randn',
    'ones', 'ones_like',
    'zeros', 'zeros_like',
    'nn_resize', 'bilinear_resize',
    'roi_pool', 'roi_align',
]


##############################################
#                                            #
#                Arithmetic                  #
#                                            #
##############################################


def _fundamental(input, value, op='Add', out=None):
    if not isinstance(value, Tensor):
        value = WrapScalar(value, input.dtype, input.device)
    dev = MakeDevice(inputs=[input, value])
    key = '{}/{}'.format(op, dev)
    module = get_module(Fundamental, key, dev, op_type=op)
    return module.forward(input, value, out)


def _rfundamental(input, value, op='RAdd', out=None):
    if not isinstance(value, Tensor):
        value = WrapScalar(value, input.dtype, input.device)
    dev = MakeDevice(inputs=[input, value])
    key = '{}/{}'.format(op, dev)
    module = get_module(Fundamental, key, dev, op_type=op)
    return module.forward(value, input, out)


def add(input, value, out=None):
    """Add the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : dragon.vm.torch.Tensor, number
        The value tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Add')


def sub(input, value, out=None):
    """Subtract the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : dragon.vm.torch.Tensor or number
        The value tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Sub')


def mul(input, value, out=None):
    """Multiply the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : dragon.vm.torch.Tensor or number
        The value tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Mul')


def div(input, value, out=None):
    """Divide the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : dragon.vm.torch.Tensor or number
        The value tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Div')


def maximum(input, other, out=None):
    """Return the max value of given two tensors.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor or number
        The input tensor.
    other : dragon.vm.torch.Tensor or number
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not isinstance(input, Tensor):
        input = WrapScalar(input, other.dtype, other.device)
    elif not isinstance(other, Tensor):
        other = WrapScalar(other, input.dtype, input.device)
    dev = MakeDevice(inputs=[input])
    key = 'Maximum/{}'.format(dev)
    module = get_module(Maximum, key, dev)
    return module.forward(input, other, out)


def minimum(input, other, out=None):
    """Return the min value of given two tensors.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor or number
        The input tensor.
    other : dragon.vm.torch.Tensor or number
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not isinstance(input, Tensor):
        input = WrapScalar(input, other.dtype, other.device)
    elif not isinstance(other, Tensor):
        other = WrapScalar(other, input.dtype, input.device)
    dev = MakeDevice(inputs=[input])
    key = 'Minimum/{}'.format(dev)
    module = get_module(Minimum, key, dev)
    return module.forward(input, other, out)


def clamp(input, min=None, max=None, out=None):
    """Clamp all elements into the range [min, max].

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    min : number, optional
        The min value.
    max : number, optional
        The max value.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Clamp/{}/min:{}/max:{}'.format(dev, min, max)
    module = get_module(Clamp, key, dev, min=min, max=max)
    return module.forward(input, out)


def log(input, out=None):
    """Compute the natural logarithm of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Log/{}'.format(dev)
    module = get_module(Log, key, dev)
    return module.forward(input, out)


def exp(input, out=None):
    """Compute the exponential of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Exp/{}'.format(dev)
    module = get_module(Exp, key, dev)
    return module.forward(input, out)


def sqrt(input, out=None):
    """Compute the square-root of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Sqrt/{}'.format(dev)
    module = get_module(Sqrt, key, dev)
    return module.forward(input, out)


def accumulate(input, alpha=1., beta=1., out=None):
    """Compute *out = alpha * input + beta * out*

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    alpha : float, optional, default=1.
        The value of alpha.
    beta : float, optional, default=1.
        The value beta.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Accumulate/{}/alpha:{}/beta:{}'.format(dev, alpha, beta)
    module = get_module(Accumulate, key, dev, alpha=alpha, beta=beta)
    return module.forward(input, out)


def mm(mat1, mat2, transA=False, transB=False, out=None):
    """Performs a matrix multiplication of the matrices ``mat1`` and ``mat2.``

    Parameters
    ----------
    mat1 : dragon.vm.torch.Tensor
        The matrix A.
    mat2 : dragon.vm.torch.Tensor
        The matrix B.
    transA : boolean
        Whether to transpose the ``mat1``.
    transB : boolean
        Whether to transpose the ``mat2``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[mat1, mat2])
    key = 'Matmul/{}/transA:{}/transB:{}'.format(dev, transA, transB)
    module = get_module(MM, key, dev, transA=transA, transB=transB)
    return module.forward(mat1, mat2, out)


def xw_plus_b(x, w, bias=None, transW=True, out=None):
    """Compute *matmul(x, w) + bias.*``

    Parameters
    ----------
    x : dragon.vm.torch.Tensor
        The x.
    w : dragon.vm.torch.Tensor
        The w.
    bias : dragon.vm.torch.Tensor, optional
        The bias.
    transW : boolean
        Whether to transpose the ``w``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[x, w] + ([bias] if bias else []))
    key = 'FullyConnected/{}/transW:{}'.format(dev, transW)
    module = get_module(FullyConnected, key, dev, transW=transW)
    return module.forward(x, w, bias, out)


##############################################
#                                            #
#                   Array                    #
#                                            #
##############################################


def _reshape(input, shape, shape_like=None):
    if shape_like is not None: shape = shape_like.shape
    dev = MakeDevice(inputs=[input]); ndim = len(shape)
    key = 'Reshape/{}/ndim:{}'.format(dev, ndim)
    module = get_module(Reshape, key, dev, ndim=ndim)
    return module.forward(input, shape)


def _permute(input, perm):
    dev = MakeDevice(inputs=[input]); nperm = len(perm)
    key = 'Permute/{}/nperm:{}'.format(dev, nperm)
    module = get_module(Permute, key, dev, nperm=nperm)
    return module.forward(input, perm)


def _repeat(input, times):
    dev = MakeDevice(inputs=[input]); ntimes = len(times)
    key = 'Repeat/{}/ntimes:{}'.format(dev, ntimes)
    module = get_module(Repeat, key, dev, ntimes=ntimes)
    return module.forward(input, times)


def _fill(input, shape, value):
    dev = MakeDevice(inputs=[input]); ndim = len(shape)
    key = 'Fill/{}/dtype:{}/ndim:{}/value:{}' \
        .format(dev, input.dtype, ndim, value)
    module = get_module(
        Fill, key, dev,
        ndim=ndim,
        value=value,
        dtype=input.dtype,
    )
    return module.forward(input, shape)


def _uniform(input, shape, low, high):
    dev = MakeDevice(inputs=[input]); ndim = len(shape)
    key = 'Uniform/{}/dtype:{}/ndim:{}/low:{}/high:{}'.format(
        dev, input.dtype, ndim, float(low), float(high))
    module = get_module(
        RandomUniform, key, dev,
        ndim=ndim,
        low=low,
        high=high,
        dtype=input.dtype,
    )
    return module.forward(input, shape)


def _normal(input, shape, mean, std):
    dev = MakeDevice(inputs=[input]); ndim = len(shape)
    key = 'Normal/{}/dtype:{}/ndim:{}/mean:{}/std:{}'.format(
        dev, input.dtype, ndim, float(mean), float(std))
    module = get_module(
        RandomNormal, key, dev,
        ndim=ndim,
        mean=mean,
        std=std,
        dtype=input.dtype,
    )
    return module.forward(input, shape)


def _reduce(input, operation, dim=None, keepdim=False, out=None):
    if dim is None: keepdim = False
    dev = MakeDevice(inputs=[input])
    key = '{}/{}/dim:{}/keepdim:{}'.format(
        operation, dev, dim, int(keepdim))
    module = get_module(
        Reduce, key, dev,
        dim=dim,
        keepdim=keepdim,
        operation=operation,
    )
    return module.forward(input, out)


def _arg_reduce(input, operation, dim=None, keepdim=False, topk=1, out=None):
    if dim is None: keepdim = False
    dev = MakeDevice(inputs=[input])
    key = '{}/{}/dim:{}/keepdim:{}/topk:{}'.format(
        operation, dev, dim, int(keepdim), topk)
    module = get_module(
        ArgReduce, key, dev,
        axis=dim,
        topk=topk,
        keepdim=keepdim,
        operation=operation,
    )
    return module.forward(input, out)


def _index(input, starts, sizes):
    nstarts, nsizes = len(starts), len(sizes)
    dev = MakeDevice(inputs=[input])
    key = 'Index/{}/nstarts:{}/nsizes:{}'.format(dev, nstarts, nsizes)
    module = get_module(Indexing, key, dev, nstarts=nstarts, nsizes=nsizes)
    return module.forward(input, starts, sizes)


def _assign(output, starts, sizes, input):
    if not isinstance(input, Tensor):
        if isinstance(input, (tuple, list)):
            input = Tensor(input, dtype=output.dtype, device=output.device)
        else:
            input = WrapScalar(input, output.dtype, output.device)
    nstarts, nsizes = len(starts), len(sizes)
    dev = MakeDevice(inputs=[input])
    key = 'Assign/{}/nstarts:{}/nsizes:{}'.format(dev, nstarts, nsizes)
    module = get_module(Assign, key, dev, nstarts=nstarts, nsizes=nsizes)
    return module.forward(input, output, starts, sizes)


def where(condition, x, y):
    """Select elements from either ``x`` or ``y``, depending on ``condition``.

    Parameters
    ----------
    condition : dragon.vm.torch.Tensor
        The byte condition tensor.
    x : dragon.vm.torch.Tensor
        The elements for *1*.
    y : dragon.vm.torch.Tensor
        The elements for *0*.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[condition, x, y])
    key = 'Where/{}'.format(dev)
    module = get_module(Where, key, dev)
    return module.forward(condition, x, y)


def _masked_assign(output, mask, input):
    if not isinstance(input, Tensor):
        if isinstance(input, (tuple, list)):
            input = Tensor(input, dtype=output.dtype, device=output.device)
        else:
            input = WrapScalar(input, output.dtype, output.device)
    dev = MakeDevice(inputs=[input])
    key = 'MaskedAssign/{}'.format(dev)
    module = get_module(MaskedAssign, key, dev)
    return module.forward(input, output, mask)


def _compare(input, other, operation, out=None):
    if not isinstance(other, Tensor):
        other = WrapScalar(other, input.dtype, input.device)
    dev = MakeDevice(inputs=[input, other])
    key = 'Compare/{}/{}'.format(operation, dev)
    module = get_module(Compare, key, dev, operation=operation)
    return module.forward(input, other, out)


def squeeze(input, dim=None, out=None):
    """Return a tensor with all the dimensions of input of size 1 removed.

    Parameters
    ----------
    dim : int
        The optional dim to remove.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The new tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Squeeze/{}/dim:{}'.format(dev, dim if dim else 'None')
    module = get_module(Squeeze, key, dev, dim=dim)
    return module.forward(input, out=out)


def unsqueeze(input, dim, out=None):
    """Return a tensor with a dimension of size 1 inserted at the specified position.

    Parameters
    ----------
    dim : int
        The dim to remove.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The new tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Unsqueeze/{}/dim:{}'.format(dev, dim if dim else 'None')
    module = get_module(UnSqueeze, key, dev, dim=dim)
    return module.forward(input, out=out)


def mean(input, dim=None, keepdim=False, out=None):
    """Return the mean of all elements or elements along the given dim.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The axis of tensor to compute mean value.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The mean-reduced tensor.

    """
    return _reduce(input, 'MEAN', dim, keepdim, out)


def sum(input, dim=None, keepdim=False, out=None):
    """Return the sum of all elements or elements along the given dim.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The axis of tensor to compute sum value.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    torch.Tensor
        The sum-reduced tensor.

    """
    return _reduce(input, 'SUM', dim, keepdim, out)


def argmax(input, dim=None, keepdim=False, out=None):
    """Return the indices of maximum elements along the given axis.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The axis of tensor to compute sum value.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    torch.Tensor
        The maximum indices.

    """
    return _arg_reduce(input, 'ARGMAX', dim, keepdim, 1, out)


def max(input, dim=None, keepdim=False, out=None):
    """Return the values and indices of maximum elements along the given axis.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The axis of tensor to compute sum value.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not.
    out : dragon.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    tuple
        The maximum values and indices.

    """
    return _arg_reduce(input, 'MAX', dim, keepdim, 1, out)


def argmin(input, dim=None, keepdim=False, out=None):
    """Return the indices of minimum elements along the given axis.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The axis of tensor to compute sum value.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    torch.Tensor
        The minimum indices.

    """
    return _arg_reduce(input, 'ARGMIN', dim, keepdim, 1, out)


def min(input, dim=None, keepdim=False, out=None):
    """Return the values and indices of maximum elements along the given axis.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The axis of tensor to compute sum value.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not.
    out : dragon.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    sequence
        The minimum values and indices.

    """
    return _arg_reduce(input, 'MIN', dim, keepdim, 1, out)


def topk(input, k, dim=None, largest=True, sorted=True, out=None):
    """Return the k largest/smallest values and indices along the given axis.

    If ``dim`` is not given, the last dimension of the input is chosen.

    If ``largest`` is False then the k smallest elements are returned.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    k : int
        The top k.
    dim : int, optional
        The axis of tensor to compute sum value.
    largest : bool, optional
        Whether to return largest or smallest elements.
    sorted : bool, optional
        Whether to return in the sorted order.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    sequence
        The values and indices.

    """
    operation = 'MAX' if largest else 'MIN'
    if dim is None: dim = input.ndimension() - 1
    return _arg_reduce(input, operation, dim, True, k, out)


def gt(input, other, out=None):
    """Compute *input* > *other* element-wise.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor, number
        The other tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _compare(input, other, 'GT', out)


def ge(input, other, out=None):
    """Compute *input* >= *other* element-wise.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor, number
        The other tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _compare(input, other, 'GE', out)


def lt(input, other, out=None):
    """Compute *input* < *other* element-wise.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor, number
        The other tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _compare(input, other, 'LT', out)


def le(input, other, out=None):
    """Compute *input* <= *other* element-wise.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor, number
        The other tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _compare(input, other, 'LE', out)


def eq(input, other, out=None):
    """Compute *input* == *other* element-wise.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor, number
        The other tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _compare(input, other, 'EQ', out)


def ne(input, other, out=None):
    """Compute *input* != *other* element-wise.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor, number
        The other tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _compare(input, other, 'NE', out)


def cat(seq, dim=0, out=None):
    """Concatenate the inputs along the given axis.

    Parameters
    ----------
    seq : sequence of dragon.vm.torch.Tensor
        The sequence.
    dim : int, optional
        The dim to concatenate.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=seq, outputs=[out] if out else [])
    key = 'Concat/{}/dim:{}'.format(dev, dim)
    module = get_module(Concat, key, dev, axis=dim)
    return module.forward(seq, out)


def stack(seq, dim=0, out=None):
    """Stack the inputs along the given axis.

    Parameters
    ----------
    seq : sequence of dragon.vm.torch.Tensor
        The sequence.
    dim : int, optional
        The dim to stack.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(seq, [out] if out else [])
    key = 'Stack/{}/dim:{}'.format(dev, dim)
    module = get_module(Stack, key, dev, axis=dim)
    return module.forward(seq, out)


def index_select(input, dim, index, out=None):
    """Select the input values along the given axis using index.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The values.
    dim : int
        The dim to gather.
    index : dragon.vm.torch.Tensor
        The indices.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice([input, index], [out] if out else [])
    key = 'IndexSelect/{}/dim:{}'.format(dev, dim)
    module = get_module(IndexSelect, key, dev, axis=dim)
    return module.forward(input, index, out)


def masked_select(input, mask, out=None):
    """Select the input values where mask is *1*.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The values.
    mask : dragon.vm.torch.Tensor
        The mask to select values.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice([input, mask], [out] if out else [])
    key = 'MaskedSelect/{}'.format(dev)
    module = get_module(MaskedSelect, key, dev)
    return module.forward(input, mask, out)


def narrow(input, dimension, start, length):
    """Return a new tensor that is a narrowed version of input tensor.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    dimension : int
        The dimension to narrow.
    start : int
        The starting position.
    length : int
        The distance to the ending postion.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    sizes = list(input.shape[:]); starts = [0] * len(sizes)
    starts[dimension], sizes[dimension] = start, length
    return _index(input, starts, sizes)


def nonzero(input, out=None):
    """Return the indices of non-zero elements.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.FloatTensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'NonZero/{}'.format(dev)
    module = get_module(NonZero, key, dev)
    return module.forward(input, out)


def one_hot(input, depth):
    """Return a ont hot tensor according to given input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    depth : int
        The depth of channels.

    Returns
    -------
    dragon.vm.torch.FloatTensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'OneHot/{}/depth:{}'.format(dev, depth)
    module = get_module(OneHot, key, dev, depth=depth)
    return module.forward(input)


def multinomial(input, num_samples, eps=0., out=None):
    """Return a tensor where each row contains ``num_samples``,
     sampled from the multinomial distribution.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    num_samples : int
        The number of samples.
    eps : float, optional, default=0.
        The prob to a uniform sampling.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dev = MakeDevice(inputs=[input])
    key = 'Multinomial/{}' \
          '/num_samples:{}' \
          '/eps:{}'.format(dev, num_samples, eps)
    module = get_module(
        Multinomial, key, dev,
        eps=eps,
        num_samples=num_samples,
    )
    return module.forward(input, out)


##############################################
#                                            #
#                 Creation                   #
#                                            #
##############################################


def _get_leaf_tensor(sizes, kwargs):
    return _LeafTensor(sizes,
        requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False,
        dtype=kwargs.get('dtype', 'float32'),
        device=kwargs.get('device', _Device()))


def zeros(*sizes, **kwargs):
    """Return a float tensor with values of ``0``.

    Parameters
    ----------
    sizes : tuple, list or int
        The sizes indicating the shape of the output tensor.
    out : dragon.vm.torch.Tensor
        The optional output tensor.

    Returns
    -------
    vm.torch.FloatTensor
        The output tensor.

    """
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None: out = _get_leaf_tensor(sizes, kwargs)
    return _fill(out, shape=sizes, value=0)


def zeros_like(input, out=None, **kwargs):
    """Return a float tensor with values of ``0``, shape as the input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating shape.
    out : dragon.vm.torch.Tensor
        The optional output tensor.

    Returns
    -------
    vm.torch.FloatTensor
        The output tensor.

    """
    if not hasattr(input, 'shape'):
        raise ValueError('Input does not have the shape attribute.')
    if out is None: out = _get_leaf_tensor(input.shape, kwargs)
    return _fill(out, shape=input.shape, value=0)


def ones(*sizes, **kwargs):
    """Return a float tensor with values of ``1``.

    Parameters
    ----------
    sizes : tuple, list or int
        The sizes indicating the shape of the output tensor.
    out : dragon.vm.torch.Tensor
        The optional output tensor.

    Returns
    -------
    vm.torch.FloatTensor
        The output tensor.

    """
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None: out = _get_leaf_tensor(sizes, kwargs)
    return _fill(out, shape=sizes, value=1)


def ones_like(input, out=None, **kwargs):
    """Return a float tensor with values of ``1``, shape as the input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating shape.
    out : dragon.vm.torch.Tensor
        The optional output tensor.

    Returns
    -------
    vm.torch.FloatTensor
        The output tensor.

    """
    if not hasattr(input, 'shape'):
        raise ValueError('Input does not have the shape attribute.')
    if out is None: out = _get_leaf_tensor(input.shape, kwargs)
    return _fill(out, shape=input.shape, value=1)


def rand(*sizes, **kwargs):
    """Return a float tensor with a uniform distribution of U(0, 1).

    Parameters
    ----------
    sizes : tuple, list or int
        The sizes indicating the shape of the output tensor.
    out : dragon.vm.torch.Tensor
        The optional output tensor.

    Returns
    -------
    vm.torch.FloatTensor
        The output tensor.

    """
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None: out = _get_leaf_tensor(sizes, kwargs)
    return _uniform(out, sizes, low=0, high=1)


def randn(*sizes, **kwargs):
    """Return a float tensor with a normal distribution of N(0, 1).

    Parameters
    ----------
    sizes : tuple, list or int
        The sizes indicating the shape of the output tensor.
    out : dragon.vm.torch.Tensor
        The optional output tensor.

    Returns
    -------
    vm.torch.FloatTensor
        The output tensor.

    """
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None: out = _get_leaf_tensor(sizes, kwargs)
    return _normal(out, sizes, mean=0, std=1)


##############################################
#                                            #
#                  Update                    #
#                                            #
##############################################


def _accumulate(grads):
    if len(grads) == 0: return
    if not isinstance(grads, (list, tuple)): grads = [grads]
    dev = MakeDevice(inputs=grads)
    key = 'Accumulate/{}/alpha:1./beta:1.'.format(dev)
    module = get_module(_Accumulate, key, dev)
    return module.forward(grads)


def _allreduce(grads):
    if not isinstance(grads, (list, tuple)): grads = [grads]
    dev = MakeDevice(inputs=grads)
    mode = mpi.GetParallelMode() + '_ALLREDUCE'
    key = 'Collective/{}/{}'.format(dev, mode.lower())
    module = get_module(Collective, key, dev, mode=mode)
    return module.forward(grads)


def _update(
    param,
    grad,
    op_type,
    slot,
    lr_mult=1.0,
    decay_mult=1.0,
):
    dev = MakeDevice(inputs=[param])
    key = '{}/{}/{}/{}'.format(op_type, dev, slot, param.name)
    module = get_module(
        Update, key, dev,
        op_type=op_type,
        lr_mult=lr_mult,
        decay_mult=decay_mult,
        slot=slot,
    )
    return module.forward(param, grad)


##############################################
#                                            #
#                  Vision                    #
#                                            #
##############################################


def _resize_2d(input, op_type, dsize, fx, fy):
    if dsize is None:
        if fx < 0 or fy < 0:
            raise ValueError('Set fx and fy if dsize is None.')
    else:
        if len(dsize) != 2:
            raise ValueError('The dsize should be a list with 2 elements.')
    if dsize is None and (fy == -1.0 or fx == -1.0):
        raise RuntimeError('The dsize, fx/fy should be specified either.')
    dev = MakeDevice(inputs=[input])
    key = '{}/{}/dsize:{}/fx:{}/fy:{}'.format(
        op_type, dev, '2' if dsize else 'none', fx, fy)
    module = get_module(
        Resize2d, key, dev,
        dsize=dsize,
        fx=fx, fy=fy,
        op_type=op_type,
    )
    return module.forward(input, dsize)


def nn_resize(input, dsize, fx=-1.0, fy=-1.0):
    return _resize_2d(input, 'NNResize', dsize, fx, fy)


def bilinear_resize(input, dsize, fx=-1.0, fy=-1.0):
    return _resize_2d(input, 'BilinearResize', dsize, fx, fy)


def roi_pool(
    feature,
    rois,
    pooled_h,
    pooled_w,
    spatial_scale,
):
    dev = MakeDevice(inputs=[feature])
    key = 'RoIPool/{}' \
          '/pool_h:{}' \
          '/pool_w:{}' \
          '/spatial_scale:{}' \
        .format(dev,
                pooled_h,
                pooled_w,
                spatial_scale)
    module = get_module(
        RoIPool, key, dev,
        pooled_h=pooled_h,
        pooled_w=pooled_w,
        spatial_scale=spatial_scale,
    )
    return module.forward(feature, rois)


def roi_align(
    feature,
    rois,
    pooled_h,
    pooled_w,
    spatial_scale,
    sampling_ratio=2,
):
    dev = MakeDevice(inputs=[feature])
    key = 'RoIAlign/{}' \
          '/pool_h:{}' \
          '/pool_w:{}' \
          '/spatial_scale:{}' \
          '/sampling_ratio:{}' \
        .format(dev,
                pooled_h,
                pooled_w,
                spatial_scale,
                sampling_ratio)
    module = get_module(
        RoIAlign, key, dev,
        pooled_h=pooled_h,
        pooled_w=pooled_w,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
    )
    return module.forward(feature, rois)