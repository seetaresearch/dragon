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

from dragon.vm.torch.tensor import LeafTensor

from dragon.vm.torch.ops.array import (
    _fill, _uniform, _normal,
)


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
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
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
    if out is None:
        out = LeafTensor(input.shape, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
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
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
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
    if out is None:
        out = LeafTensor(input.shape, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
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
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
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
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    return _normal(out, sizes, mean=0, std=1)