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
from dragon.vm.torch.execution import RunOperator
from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.factory import get_module

from dragon.vm.torch.ops.modules.creation import OneHot


__all__= [
    'zeros', 'zeros_like', 'ones', 'ones_like',
    'one_hot', 'rand', 'randn',
]


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
    arguments = {'value': 0.0, 'dims': sizes}
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    inputs = []; outputs = [out]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'Fill', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


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
    arguments = {'value': 0.0, 'dims': input.shape}
    if out is None:
        out = LeafTensor(input.shape, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    inputs = []; outputs = [out]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'Fill', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


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
    arguments = {'value': 1.0, 'dims': sizes}
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    inputs = []; outputs = [out]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'Fill', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


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
    arguments = {'value': 1.0, 'dims': input.shape}
    if out is None:
        out = LeafTensor(input.shape, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    inputs = []; outputs = [out]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'Fill', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


def rand(*sizes, **kwargs):
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
    arguments = {'low': 0.0, 'high': 1.0, 'dims': sizes}
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    inputs = []; outputs = [out]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'RandomUniform', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


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
    arguments = {'mean': 0.0, 'std': 1.0, 'dims': sizes}
    out = kwargs['out'] if 'out' in kwargs else None
    if out is None:
        out = LeafTensor(sizes, requires_grad=kwargs['requires_grad'] \
            if 'requires_grad' in kwargs else False)
    inputs = []; outputs = [out]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'RandomNormal', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


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
    vm.torch.FloatTensor
        The output tensor.

    """
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.one_hot/{}:{}/depth:{}'.format(
        ctx[0], ctx[1], depth)
    module = get_module(OneHot, key, ctx, depth=depth)
    return module.forward(input)