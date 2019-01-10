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

from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.modules.shape import \
    Reshape, Squeeze, UnSqueeze, Fill, Permute, Repeat
from dragon.vm.torch.ops.modules.reduce import Reduce, ArgReduce
from dragon.vm.torch.ops.modules.indexing import Indexing
from dragon.vm.torch.ops.modules.axis import Concat, Gather


def reshape(input, shape, shape_like=None):
    if shape_like is not None: shape = shape_like.shape
    ctx = MakeContext(inputs=[input]); n_dim = len(shape)
    key = 'torch.ops.reshape/{}:{}/n_dim:{}'.format(ctx[0], ctx[1], n_dim)
    module = get_module(Reshape, key, ctx, n_dim=n_dim)
    return module.forward(input, shape)


def squeeze(input, dim=None, out=None):
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.squeeze/{}:{}/dim:{}'.format(
        ctx[0], ctx[1], dim if dim else 'None')
    module = get_module(Squeeze, key, ctx, dim=dim)
    return module.forward(input, out=out)


def unsqueeze(input, dim, out=None):
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.unsqueeze/{}:{}/dim:{}'.format(
        ctx[0], ctx[1], dim if dim else 'None')
    module = get_module(UnSqueeze, key, ctx, dim=dim)
    return module.forward(input, out=out)


def _permute(input, perm=None):
    ctx = MakeContext(inputs=[input]); n_perm = len(perm) if perm else 0
    key = 'torch.ops.permute/{}:{}/n_perm:{}'.format(ctx[0], ctx[1], n_perm)
    module = get_module(Permute, key, ctx, n_perm=n_perm)
    return module.forward(input, perm)


def _repeat(input, times):
    ctx = MakeContext(inputs=[input]); n_times = len(times)
    key = 'torch.ops.repeat/{}:{}/n_times:{}'.format(ctx[0], ctx[1], n_times)
    module = get_module(Repeat, key, ctx, n_times=n_times)
    return module.forward(input, times)


def _fill(input, shape, value):
    ctx = MakeContext(inputs=[input]); n_dim = len(shape)
    key = 'torch.ops.fill/{}:{}/dtype:{}/n_dim:{}/value:{}'.format(
        ctx[0], ctx[1], input._dtype, n_dim, value)
    module = get_module(Fill, key, ctx, n_dim=n_dim,
        value=value, dtype=input._dtype)
    return module.forward(input, shape)


def _reduce(input, operation, dim=None, keepdim=False, out=None):
    ctx = MakeContext(inputs=[input])
    if dim is None: keepdim = False
    key = 'torch.ops.{}/{}:{}/dim:{}/keepdim:{}'.format(operation.lower(),
        ctx[0], ctx[1], dim, int(keepdim))
    module = get_module(Reduce, key, ctx,
        operation=operation, dim=dim, keepdim=keepdim)
    return module.forward(input, out)


def _arg_reduce(input, operation, dim=None, keepdim=False, top_k=1, out=None):
    ctx = MakeContext(inputs=[input])
    if dim is None: keepdim = False
    key = 'torch.ops.{}/{}:{}/dim:{}/keepdim:{}/top_k:{}'.format(operation.lower(),
        ctx[0], ctx[1], dim, int(keepdim), top_k)
    module = get_module(ArgReduce, key, ctx, operation=operation,
        axis=dim, keepdim=keepdim, top_k=top_k)
    return module.forward(input, out)


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
    ctx = MakeContext(inputs=seq, outputs=[out] if out else [])
    key = 'torch.ops.cat/{}:{}/dim:{}'.format(
        ctx[0], ctx[1], dim)
    module = get_module(Concat, key, ctx, axis=dim)
    return module.forward(seq, out)


def gather(input, dim, index, out=None):
    """Gather the input values along the given axis.

    Note that it is a tensorflow style gather, which takes a vector index,

    values of other dimension will be copied automatically.

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
    ctx = MakeContext(inputs=[input, index], outputs=[out] if out else [])
    key = 'torch.ops.gather/{}:{}/dim:{}'.format(ctx[0], ctx[1], dim)
    module = get_module(Gather, key, ctx, axis=dim)
    return module.forward(input, index, out)


def _indexing(input, starts, sizes):
    n_starts, n_sizes = len(starts), len(sizes)
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.indexing/{}:{}/n_starts:{}/n_sizes:{}'.format(
        ctx[0], ctx[1], n_starts, n_sizes)
    module = get_module(Indexing, key, ctx, n_starts=n_starts, n_sizes=n_sizes)
    return module.forward(input, starts, sizes)


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
    return _indexing(input, starts, sizes)