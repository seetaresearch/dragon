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
from dragon.vm.torch.ops.modules.arithmetic import Fundamental


def _fundamental(input, value, op='Add', out=None):
    if not isinstance(value, Tensor):
        if not isinstance(value, (int, float)):
            raise TypeError('Type of value should be numerical, got {}.'
                    .format(type(value)))
        value = WrapScalar(value, input._dtype, input._ctx)
    ctx = MakeContext(inputs=[input, value])
    key = 'torch/ops/{}/{}:{}'.format(op.lower(), ctx[0].lower(), ctx[1])
    module = get_module(Fundamental, key, ctx, op_type=op)
    return module.forward(input, value, out)


def _rfundamental(input, value, op='RAdd', out=None):
    if not isinstance(value, Tensor):
        if not isinstance(value, (int, float)):
            raise TypeError('Type of value should be numerical, got {}.'
                    .format(type(value)))
        value = WrapScalar(value, input._dtype, input._ctx)

    ctx = MakeContext(inputs=[input, value])
    key = 'torch/ops/{}/{}:{}'.format(op.lower(), ctx[0].lower(), ctx[1])
    module = get_module(Fundamental, key, ctx, op_type=op)
    return module.forward(value, input, out)


def add(input, value, out=None):
    """Add the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    value : vm.torch Tensor, int or float
        The value tensor.
    out : vm.torch.Tensor or None
        The output tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Add')


def sub(input, value, out=None):
    """Subtract the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    value : vm.torch Tensor, int or float
        The value tensor.
    out : vm.torch.Tensor or None
        The output tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Sub')


def mul(input, value, out=None):
    """Multiply the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    value : vm.torch Tensor, int or float
        The value tensor.
    out : vm.torch.Tensor or None
        The output tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Mul')


def div(input, value, out=None):
    """Divide the ``input`` and ``value`` into the output tensor.

    Parameters
    ----------
    input : vm.torch.Tensor
        The input tensor.
    value : vm.torch Tensor, int or float
        The value tensor.
    out : vm.torch.Tensor or None
        The output tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(input, value, out=out, op='Div')