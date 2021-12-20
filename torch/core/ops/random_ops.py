# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Random ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest

from dragon.vm.torch.core import cpp
from dragon.vm.torch.core.autograd.function import Function


def normal(mean, std, size, out=None):
    r"""Return a tensor initialized from the normal distribution.

    .. math:: \text{out} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    mean : number
        The value to :math:`\mu`.
    std : number
        The value to :math:`\sigma`.
    size : Sequence[int]
        The output tensor shape.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dtype = out.dtype if out else 'float32'
    device = out.device if out else cpp.device()
    return Function.apply(
        'RandomNormal', device, [], outputs=[out],
        dtype=dtype, mean=float(mean), std=float(std),
        ndim=len(size), dims=size)


def rand(*size, out=None, dtype='float32', device=None, requires_grad=False):
    """Return a tensor from the uniform distribution of U(0, 1).

    Parameters
    ----------
    size : int...
        The shape of output tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = nest.flatten(size)
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'RandomUniform', device, [], outputs=[out],
        dtype=dtype, low=0.0, high=1.0, ndim=len(size), dims=size)
    out._requires_grad = requires_grad
    return out


def randn(*size, out=None, dtype='float32', device=None, requires_grad=False):
    """Return a tensor from the normal distribution of N(0, 1).

    Parameters
    ----------
    size : int...
        The output tensor shape.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = nest.flatten(size)
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'RandomNormal', device, [], outputs=[out],
        dtype=dtype, mean=0.0, std=1.0, ndim=len(size), dims=size)
    out._requires_grad = requires_grad
    return out


def randperm(n, out=None, dtype='int64', device=None, requires_grad=False):
    """Return a tensor with value in the permuted range.

    Specify ``n`` to determine an interval :math:`[0, n)`:

    ```python
    print(torch.randperm(4))
    ```

    Parameters
    ----------
    n: number
        The end of interval.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='int64'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Permutation', device, [], outputs=[out],
        dtype=dtype, limit=n)
    out._requires_grad = requires_grad
    return out


def uniform(low, high, size, out=None):
    r"""Return a tensor initialized from the uniform distribution.

    .. math:: \text{out} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    low : number
        The value to :math:`\alpha`.
    high : number
        The value to :math:`\beta`.
    size : Sequence[int]
        The output tensor shape.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dtype = out.dtype if out else 'float32'
    device = out.device if out else cpp.device()
    return Function.apply(
        'RandomUniform', device, [], outputs=[out],
        dtype=dtype, low=float(low), high=float(high),
        ndim=len(size), dims=size)
