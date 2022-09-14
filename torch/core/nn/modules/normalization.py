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
"""Normalization modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from dragon.core.util import nest
from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops import constant_ops
from dragon.vm.torch.core.tensor import Tensor


class GroupNorm(Module):
    r"""Apply the group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    Examples:

    ```python
    m = torch.nn.GroupNorm(num_groups=2, num_channels=4)
    x = torch.randn(2, 4)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.group_norm(...)`_

    """

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
    ):
        r"""Create a ``GroupNorm`` module.

        Parameters
        ----------
        num_groups : int
            The number of groups.
        num_channels : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.

        """
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(Tensor(num_channels))
            self.bias = Parameter(Tensor(num_channels))
        else:
            self.register_buffer('weight', constant_ops.ones(num_channels))
            self.register_buffer('bias', constant_ops.zeros(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.one_()
            self.bias.data.zero_()

    def extra_repr(self):
        return '{num_groups},' \
               '{num_channels}, ' \
               'eps={eps}, ' \
               'affine={affine}' \
               .format(**self.__dict__)

    def forward(self, input):
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def _apply(self, fn):
        lambda_source = inspect.getsource(fn)
        if 'half_()' in lambda_source:
            return self  # Float32 parameters are required.
        return super(GroupNorm, self)._apply(fn)


class LayerNorm(Module):
    r"""Apply the layer normalization.
    `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    Examples:

    ```python
    x = torch.randn(2, 3, 4)
    m = torch.nn.LayerNorm(x.size()[1:])
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.layer_norm(...)`_

    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        r"""Create a ``LayerNorm`` module.

        Parameters
        ----------
        normalized_shape : Union[int, Sequence[int]]
            The size normalized over the last dimensions.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        elementwise_affine : bool, optional, default=True
            ``True`` to apply an affine transformation.

        """
        super(LayerNorm, self).__init__()
        self.normalized_shape = tuple(nest.flatten(normalized_shape))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(Tensor(*self.normalized_shape))
            self.bias = Parameter(Tensor(*self.normalized_shape))
        else:
            self.register_buffer('weight', constant_ops.ones(*self.normalized_shape))
            self.register_buffer('bias', constant_ops.zeros(*self.normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight.data.one_()
            self.bias.data.zero_()

    def extra_repr(self):
        return '{normalized_shape}, ' \
               'eps={eps}, ' \
               'elementwise_affine={elementwise_affine}' \
               .format(**self.__dict__)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def _apply(self, fn):
        lambda_source = inspect.getsource(fn)
        if 'half_()' in lambda_source:
            return self  # Float32 parameters are required.
        return super(LayerNorm, self)._apply(fn)


class LocalResponseNorm(Module):
    r"""Apply the local response normalization.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    The normalization is defined as:

    .. math::
        y_{i} = x_{i}\left(k + \frac{\alpha}{n}
                     \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}x_{j}^2
                     \right)^{-\beta}

    Examples:

    ```python
    m = torch.nn.LocalResponseNorm(5)
    x = torch.randn(2, 4)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.local_response_norm(...)`_

    """

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1.):
        r"""Create a ``GroupNorm`` module.

        Parameters
        ----------
        size : int, required
            The number of neighbouring channels to sum over.
        alpha : float, optional, default=0.0001
            The value to :math:`\alpha`.
        beta : float, optional, default=0.75
            The value to :math:`\beta`.
        k : float, optional, default=1.
            The value to :math:`k`.

        """
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)

    def forward(self, input):
        return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)
