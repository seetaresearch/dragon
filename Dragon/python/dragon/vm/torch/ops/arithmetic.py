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

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.ops.primitive import MakeContext, WrapScalar
from dragon.vm.torch.ops.factory import get_module

from dragon.vm.torch.ops.modules.arithmetic import (
    Fundamental, Log, Exp,
    Maximum, Minimum, Clamp,
)


def _fundamental(input, value, op='Add', out=None):
    if not isinstance(value, Tensor):
        value = WrapScalar(value, input.dtype, input._ctx)
    ctx = MakeContext(inputs=[input, value])
    key = 'torch.ops.{}/{}:{}'.format(op.lower(), ctx[0], ctx[1])
    module = get_module(Fundamental, key, ctx, op_type=op)
    return module.forward(input, value, out)


def _rfundamental(input, value, op='RAdd', out=None):
    if not isinstance(value, Tensor):
        value = WrapScalar(value, input.dtype, input._ctx)
    ctx = MakeContext(inputs=[input, value])
    key = 'torch.ops.{}/{}:{}'.format(op.lower(), ctx[0], ctx[1])
    module = get_module(Fundamental, key, ctx, op_type=op)
    return module.forward(value, input, out)


def _maximum(input, other, out=None):
    if not isinstance(input, Tensor):
        input = WrapScalar(input, other.dtype, other._ctx)
    elif not isinstance(other, Tensor):
        other = WrapScalar(other, input.dtype, input._ctx)
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.maximum/{}:{}'.format(ctx[0], ctx[1])
    module = get_module(Maximum, key, ctx)
    return module.forward(input, other, out)


def _minimum(input, other, out=None):
    if not isinstance(input, Tensor):
        input = WrapScalar(input, other.dtype, other._ctx)
    elif not isinstance(other, Tensor):
        other = WrapScalar(other, input.dtype, input._ctx)
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.minimum/{}:{}'.format(ctx[0], ctx[1])
    module = get_module(Minimum, key, ctx)
    return module.forward(input, other, out)


def _clamp(input, min=None, max=None, out=None):
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.clamp/{}:{}/min:{}/max:{}'.format(
        ctx[0], ctx[1], min, max)
    module = get_module(Clamp, key, ctx, min=min, max=max)
    return module.forward(input, out)


def _exp(input, out=None):
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.exp/{}:{}'.format(ctx[0], ctx[1])
    module = get_module(Exp, key, ctx)
    return module.forward(input, out)


def _log(input, out=None):
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.log/{}:{}'.format(ctx[0], ctx[1])
    module = get_module(Log, key, ctx)
    return module.forward(input, out)


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
    return _maximum(input, other, out)


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
    return _minimum(input, other, out)


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
    return _clamp(input, min, max, out)


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
    return _log(input, out)


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
    return _exp(input, out)