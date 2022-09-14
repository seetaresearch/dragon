# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py>
#
# ------------------------------------------------------------
"""NN init functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from dragon.core.util import math_util
from dragon.vm.torch.core.autograd import grad_mode
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.ops import constant_ops


def calculate_gain(nonlinearity, param=None):
    """Return the gain value according to nonlinearity function.

    Parameters
    ----------
    nonlinearity : str
        The nonlinearity to compute gain value.
    param : number, optional
        The optional param value for nonlinearity.

    Returns
    -------
    number
        The gain value.

    """
    linear_fns = [
        'linear',
        'conv1d',
        'conv2d',
        'conv3d',
        'conv_transpose1d',
        'conv_transpose2d',
        'conv_transpose3d',
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        else:
            try:
                negative_slope = float(param)
            except ValueError:
                raise ValueError("Negative slope {} is not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError('Unsupported nonlinearity: ' + nonlinearity)


def constant_(tensor, val):
    r"""Fill tensor with the scalar value.

    .. math:: \text{tensor} \leftarrow \text{value}

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    val : number
        The value to fill.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    with grad_mode.no_grad():
        return tensor.fill_(val)


def dirac_(tensor, groups=1):
    """Fill tensor with the dirac delta function.

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    groups : number, optional, default=1
        The groups of convolution.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError('Only tensors with 3, 4, or 5 dimensions are supported.')
    sizes = tensor.size()
    if sizes[0] % groups != 0:
        raise ValueError('Dimension 0 should be divisible by groups.')
    out_channels_per_grp = sizes[0] // groups
    min_dim = min(out_channels_per_grp, sizes[1])
    with grad_mode.no_grad():
        tensor.zero_()
        for g in range(groups):
            for d in range(min_dim):
                item = [g * out_channels_per_grp + d, d]
                for i in range(2, dimensions):
                    item.append(sizes[i] // 2)
                tensor[tuple(item)] = 1
        return tensor


def eye_(tensor):
    r"""Fill tensor as the identity matrix.

    .. math:: \text{tensor} \leftarrow \text{diag}(1, 1, ..., 1)

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if tensor.ndimension() != 2:
        raise ValueError('Only tensors with 2 dimensions are supported.')
    with grad_mode.no_grad():
        constant_ops.eye(*tensor.shape, out=tensor,
                         requires_grad=tensor.requires_grad)
    return tensor


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fill tensor from a kaiming normal distribution.

    .. math::
        \text{tensor} \sim \mathcal{N}(0, \sigma^{2}) \\ \, \\ \,
            \text{where} \quad \sigma = \text{gain} \times
                \sqrt{\frac{1}{\text{fan}}}

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    a : number, optional, default=0
        The negative slope to compute gain value.
    mode : {'fan_in', 'fan_out'}, optional
        The mode to compute fans.
    nonlinearity : str, optional, default='leaky_relu'
        The nonlinearity to compute gain value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with grad_mode.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fill tensor from a kaiming uniform distribution.

    .. math::
        \text{tensor} \sim \mathcal{U}(-\alpha, \alpha) \\ \, \\ \,
            \text{where} \quad \alpha = \text{gain} \times
                \sqrt{\frac{3}{\text{fan}}}

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    a : number, optional, default=0
        The negative slope to compute gain value.
    mode : {'fan_in', 'fan_out'}, optional
        The mode to compute fans.
    nonlinearity : str, optional, default='leaky_relu'
        The nonlinearity to compute gain value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with grad_mode.no_grad():
        return tensor.uniform_(-bound, bound)


def normal_(tensor, mean=0, std=1):
    r"""Fill tensor from a normal distribution.

    .. math:: \text{tensor} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    with grad_mode.no_grad():
        return tensor.normal_(mean, std)


def ones_(tensor):
    r"""Fill tensor with ones.

    .. math:: \text{tensor} \leftarrow 1

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    with grad_mode.no_grad():
        return tensor.fill_(1)


def normal_(tensor, mean=0, std=1):
    r"""Fill tensor from a normal distribution.

    .. math:: \text{tensor} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    with grad_mode.no_grad():
        return tensor.normal_(mean, std)


def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
    r"""Fill tensor from a truncated normal distribution.

    .. math:: \text{tensor} \sim \mathcal{TN}(\mu, \sigma^{2}, a, b)

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.
    a : number, optional, default=-2
        The value to :math:`a`.
    b : number, optional, default=2
        The value to :math:`b`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = tensor.size()
    with grad_mode.no_grad():
        return Function.apply(
            'TruncatedNormal', tensor.device, [], outputs=[tensor],
            dtype=tensor.dtype, mean=float(mean), std=float(std),
            low=float(a), high=float(b), ndim=len(size), dims=size)


def uniform_(tensor, a=0, b=1):
    r"""Fill tensor from an uniform distribution.

    .. math:: \text{tensor} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    a : number, optional, default=0
        The value to :math:`\alpha`.
    b : number, optional, default=1
        The value to :math:`\beta`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    with grad_mode.no_grad():
        return tensor.uniform_(a, b)


def xavier_normal_(tensor, gain=1):
    r"""Fill tensor from a xavier normal distribution.

    .. math::
        \text{tensor} \sim \mathcal{N}(0, \sigma^{2}) \\ \, \\ \,
            \text{where} \quad \sigma = \text{gain} \times
                \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    gain : number, optional, default=1
        The gain value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The input tensor.

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with grad_mode.no_grad():
        return tensor.normal_(0, std)


def xavier_uniform_(tensor, gain=1):
    r"""Fill tensor from a xavier uniform distribution.

    .. math::
        \text{tensor} \sim \mathcal{U}(-\alpha, \alpha) \\ \, \\ \,
            \text{where} \quad \alpha = \text{gain} \times
                \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    gain : number, optional, default=1
        The gain value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The input tensor.

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    with grad_mode.no_grad():
        return tensor.uniform_(-a, a)


def zeros_(tensor):
    r"""Fill tensor with zeros.

    .. math:: \text{tensor} \leftarrow 0

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    with grad_mode.no_grad():
        return tensor.fill_(0)


def _calculate_fan_in_and_fan_out(tensor):
    """Return the fan value according to tensor size."""
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Excepted 2 or higher tensor dimensions.")
    if dimensions == 2:
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input = tensor.size(1)
        num_output = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = math_util.prod(tensor.shape[2:])
        fan_in = num_input * receptive_field_size
        fan_out = num_output * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    """Return the fan value according to mode and tensor size."""
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError('Mode {} not supported, please use one of {}'.format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out
