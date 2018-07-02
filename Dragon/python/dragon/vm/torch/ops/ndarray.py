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

from dragon.vm.torch.ops.primitive import MakeContext, CanonicalAxis
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.modules.shape import Reshape, Fill, Permute, Repeat
from dragon.vm.torch.ops.modules.reduce import Reduce, ArgReduce
from dragon.vm.torch.ops.modules.crop import Crop
from dragon.vm.torch.ops.modules.axis import Concat, Gather


def reshape(input, shape, shape_like=None):
    if shape_like is not None: shape = shape_like.shape
    ctx = MakeContext(inputs=[input]); len_shape = len(shape)
    key = 'torch/ops/reshape/{}:{}/n_dims:#{}'.format(ctx[0].lower(), ctx[1], len_shape)
    module = get_module(Reshape, key, ctx, len_shape=len_shape)
    return module.forward(input, shape)


def _permute(input, perms=None):
    ctx = MakeContext(inputs=[input]); len_perms = len(perms) if perms else 0
    key = 'torch/ops/permute/{}:{}/n_dims:#{}'.format(ctx[0].lower(), ctx[1], len_perms)
    module = get_module(Permute, key, ctx, len_perms=len_perms)
    return module.forward(input, perms)


def _repeat(input, times):
    ctx = MakeContext(inputs=[input]); len_times = len(times)
    key = 'torch/ops/repeat/{}:{}/n_times:#{}'.format(ctx[0].lower(), ctx[1], len_times)
    module = get_module(Repeat, key, ctx, len_times=len_times)
    return module.forward(input, times)


def _fill(input, shape, value):
    ctx = MakeContext(inputs=[input]); len_shape = len(shape)
    key = 'torch/ops/fill/{}:{}/ndims:#{}/value:{}'.format(
        ctx[0].lower(), ctx[1], len_shape, value)
    module = get_module(Fill, key, ctx, len_shape=len_shape, value=value)
    return module.forward(input, shape)


def _reduce(input, operation, dim=None, keepdim=False, out=None):
    ctx = MakeContext(inputs=[input])
    if dim is None: dim = -1; keepdim = False
    elif dim < 0: dim = CanonicalAxis(input, dim)
    key = 'torch/ops/{}/{}:{}/dim[{}]/keep_dims:{}'.format(operation.lower(),
        ctx[0].lower(), ctx[1], dim, int(keepdim))
    module = get_module(Reduce, key, ctx,
        operation=operation, axis=dim, keep_dims=keepdim)
    return module.forward(input, out)


def _arg_reduce(input, operation, dim=None, keepdim=False, top_k=1, out=None):
    ctx = MakeContext(inputs=[input])
    if dim is None: dim = -1; keepdim = False
    elif dim < 0: dim = CanonicalAxis(input, dim)
    key = 'torch/ops/{}/{}:{}/dim[{}]/keep_dims:{}/top_k:{}'.format(operation.lower(),
        ctx[0].lower(), ctx[1], dim, int(keepdim), top_k)
    module = get_module(ArgReduce, key, ctx, operation=operation,
        axis=dim, keep_dims=keepdim, top_k=top_k)
    return module.forward(input, out)


def mean(input, dim=None, keepdim=False, out=None):
    """Return the mean of all elements or elements along the given dim.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    dim : int or None
        The axis of tensor to compute mean value.
    keepdim : boolean
        Whether the output tensor has dim retained or not.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    vm.torch.Tensor
        The mean-reduced tensor.

    """
    return _reduce(input, 'MEAN', dim, keepdim, out)


def sum(input, dim=None, keepdim=False, out=None):
    """Return the sum of all elements or elements along the given dim.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    dim : int or None
        The axis of tensor to compute sum value.
    keepdim : boolean
        Whether the output tensor has dim retained or not.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    vm.torch.Tensor
        The sum-reduced tensor.

    """
    return _reduce(input, 'SUM', dim, keepdim, out)


def argmax(input, dim=None, keepdim=False, out=None):
    """Return the indices of maximum elements along the given axis.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    dim : int or None
        The axis of tensor to compute sum value.
    keepdim : boolean
        Whether the output tensor has dim retained or not.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    vm.torch.Tensor
        The maximum indices.

    """
    return _arg_reduce(input, 'ARGMAX', dim, keepdim, 1, out)


def max(input, dim=None, keepdim=False, out=None):
    """Return the values and indices of maximum elements along the given axis.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    dim : int or None
        The axis of tensor to compute sum value.
    keepdim : boolean
        Whether the output tensor has dim retained or not.
    out : vm.torch.Tensor or None
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
    input : vm.torch.Tensor
        The input tensor.
    dim : int or None
        The axis of tensor to compute sum value.
    keepdim : boolean
        Whether the output tensor has dim retained or not.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    vm.torch.Tensor
        The minimum indices.

    """
    return _arg_reduce(input, 'ARGMIN', dim, keepdim, 1, out)


def min(input, dim=None, keepdim=False, out=None):
    """Return the values and indices of maximum elements along the given axis.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    dim : int or None
        The axis of tensor to compute sum value.
    keepdim : boolean
        Whether the output tensor has dim retained or not.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    tuple
        The minimum values and indices.

    """
    return _arg_reduce(input, 'MIN', dim, keepdim, 1, out)


def topk(input, k, dim=None, largest=True, sorted=True, out=None):
    """Return the k largest/smallest values and indices along the given axis.

    If ``dim`` is not given, the last dimension of the input is chosen.

    If ``largest`` is False then the k smallest elements are returned.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    k : int
        The top k.
    dim : int or None
        The axis of tensor to compute sum value.
    largest : boolean
        Whether to return largest or smallest elements.
    sorted : boolean
        Whether to return in the sorted order.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    tuple
        The values and indices.

    """
    operation = 'MAX' if largest else 'MIN'
    if dim is None: dim = input.ndimension() - 1
    return _arg_reduce(input, operation, dim, True, k, out)


def cat(seq, dim=0, out=None):
    """Concatenate the inputs along the given axis.

    Parameters
    ----------
    seq : tuple or list of vm.torch.Tensor
        The sequence.
    dim : int
        The dim to concatenate.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    ctx = MakeContext(inputs=seq, outputs=[out] if out else [])
    key = 'torch/ops/cat/{}:{}/dim:{}'.format(
        ctx[0].lower(), ctx[1], dim)
    module = get_module(Concat, key, ctx, axis=dim)
    return module.forward(seq, out)


def gather(input, dim, index, out=None):
    """Gather the input values along the given axis.

    Note that it is a tensorflow style gather, which takes a vector index,

    values of other dimension will be copied automatically.

    Parameters
    ----------
    input : vm.torch.Tensor
        The values.
    dim : int
        The dim to gather.
    index : vm.torch.Tensor
        The indices.
    out : vm.torch.Tensor or None
        The optional output tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    ctx = MakeContext(inputs=[input, index], outputs=[out] if out else [])
    key = 'torch/ops/gather/{}:{}/dim:{}'.format(
        ctx[0].lower(), ctx[1], dim)
    module = get_module(Gather, key, ctx, axis=dim)
    return module.forward(input, index, out)


def _crop(input, starts, ends):
    len_starts, len_ends = len(starts), len(ends)
    ctx = MakeContext(inputs=[input])
    key = 'torch/ops/crop/{}:{}/starts:#{}/ends:#{}'.format(
        ctx[0].lower(), ctx[1], len_starts, len_ends)
    module = get_module(Crop, key, ctx, len_starts=len_starts, len_ends=len_ends)
    return module.forward(input, starts, ends)