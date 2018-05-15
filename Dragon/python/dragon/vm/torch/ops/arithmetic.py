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

import numpy as np
import dragon as dg

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.tensor_uitls import from_dragon
from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.modules.arithmetic import Fundamental


def _wrap_scalar(scalar, dtype):
    # TODO(PhyscalX): We use (dtype/value) to hash different scalars.
    # TODO(PhyscalX): In Dragon, set a Tensor with same dtype and shape will not deconstruct it.
    value = np.array([scalar], dtype=dtype)
    t = dg.Tensor('/share/scalar/{}/{}'.format(
        dtype, str(float(scalar)))).Variable()
    t.set_value(value)
    return t.name


def _fundamental(input, value, op='Add', out=None):
    if not isinstance(value, Tensor):
        if not isinstance(value, (int, float)):
            raise TypeError('Type of value should be numerical, got {}.'
                            .format(type(value)))
        # TODO(PhyscalX): We simply use a global tensor to share all scalars
        scalar_name = _wrap_scalar(value, dtype=input._dtype)
        value = Tensor(dg_tensor=scalar_name, ctx=input._ctx, own_storage=False)

    ctx = MakeContext(inputs=[input, value])
    key = 'torch/ops/fundamental/{}:{}'.format(ctx[0].lower(), ctx[1])
    module = get_module(Fundamental, key, ctx, op_type=op)
    return module(input, value, out)


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