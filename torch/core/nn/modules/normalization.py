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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops.init import functional as init
from dragon.vm.torch.core.tensor import Tensor


class GroupNorm(Module):
    r"""Apply the group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    The normalization is defined as:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Examples:

    ```python
    m = torch.nn.GroupNorm(num_groups=2, num_channels=4)
    x = torch.randn(2, 4)
    y = m(x)
    ```

    """

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
    ):
        """Create a ``GroupNorm`` module.

        Parameters
        ----------
        num_groups : int
            The number of groups to split.
        num_channels : int
            The number of channels.
        eps : float, optional, default=1e-5
            The epsilon value.
        affine : bool, optional, default=True
            **True** to apply a affine transformation.

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
            self.register_buffer('weight', init.ones(num_channels))
            self.register_buffer('bias', init.zeros(num_channels))
        self.inputs = [self.weight, self.bias]
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
            input, *self.inputs,
            groups=self.num_groups,
            eps=self.eps
        )

    def _apply(self, fn):
        lambda_source = inspect.getsource(fn)
        if 'half_()' in lambda_source:
            return self  # Float32 parameters are required.
        return super(GroupNorm, self)._apply(fn)


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
    `torch.nn.functional.local_response_norm(...)`_ - Apply the local response normalization to input.

    """

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1.):
        r"""Create a ``GroupNorm`` module.

        Parameters
        ----------
        size : int, required
            The number of neighbouring channels to sum over.
        alpha : float, optional, default=0.0001
            The scale value :math:`\alpha`.
        beta : float, optional, default=0.75
            The exponent value :math:`\beta`.
        k : float, optional, default=1.
            The bias constant :math:`k`.

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
