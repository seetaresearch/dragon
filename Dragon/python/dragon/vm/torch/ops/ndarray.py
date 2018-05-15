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
from dragon.vm.torch.ops.modules.shape import Reshape, Fill
from dragon.vm.torch.ops.modules.reduce import Reduce
from dragon.vm.torch.ops.modules.crop import Crop


def reshape(input, shape, shape_like=None):
    if shape_like is not None: shape = shape_like.shape
    ctx = MakeContext(inputs=[input]); len_shape = len(shape)
    key = 'torch/ops/reshape/{}:{}/ndims:#{}'.format(ctx[0].lower(), ctx[1], len_shape)
    module = get_module(Reshape, key, ctx, len_shape=len_shape)
    return module.forward(input, shape)


def _fill(input, shape, value, out=None):
    ctx = MakeContext(inputs=[input]); len_shape = len(shape)
    key = 'torch/ops/fill/{}:{}/ndims:#{}/value:{}'.format(
        ctx[0].lower(), ctx[1], len_shape, value)
    module = get_module(Fill, key, ctx, len_shape=len_shape, value=value)
    return module.forward(input, shape, out)


def _reduce(input, op, tag=None, dim=None, keepdim=False, out=None):
    ctx = MakeContext(inputs=[input])
    if dim is None: dim = -1
    key = 'torch/ops/{}/{}:{}/dim[{}]/keep_dims:{}'.format(
        op.lower() + ':{}'.format(tag.lower()) if tag else '',
        ctx[0].lower(), ctx[1], dim, int(keepdim))
    module = get_module(Reduce, key, ctx, op_type=op,
        tag=tag, axis=dim, keep_dims=keepdim)
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
    return _reduce(input, 'Reduce', 'MEAN', dim, keepdim, out)


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
        The mean-reduced tensor.

    """
    return _reduce(input, 'Reduce', 'SUM', dim, keepdim, out)


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
        The max indices.

    """
    return _reduce(input, 'Argmax', None, dim, keepdim, out)


def argmin(input, dim=None, keepdim=False, out=None):
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
        The max indices.

    """
    return _reduce(input, 'Argmin', None, dim, keepdim, out)


def _crop(input, starts, ends):
    len_starts, len_ends = len(starts), len(ends)
    ctx = MakeContext(inputs=[input])
    key = 'torch/ops/crop/{}:{}/starts:#{}/ends:#{}'.format(
        ctx[0].lower(), ctx[1], len_starts, len_ends)
    module = get_module(Crop, key, ctx, len_starts=len_starts, len_ends=len_ends)
    return module.forward(input, starts, ends)