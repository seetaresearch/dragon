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

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops.array import functional as array_funcs
from dragon.vm.torch.core.ops.init import functional as init_funcs
from dragon.vm.torch.core.tensor import Tensor


class AffineChannel(Module):
    """Apply affine transformation along the channels.

    Affine is often taken as a post-processing of normalization.

    Examples:

    ```python
    m = torch.nn.AffineChannel(5)

    # Apply a 2d transformation
    x2d = torch.ones(3, 5)
    y2d = m(x2d)

    # Apply a 3d transformation
    x3d = torch.ones(3, 5, 4)
    y3d = m(x3d)

    # Apply a 4d transformation
    x4d = torch.ones(3, 5, 2, 2)
    y4d = m(x4d)
    ```

    See Also
    --------
    `torch.channel_affine(...)`_

    """

    def __init__(
        self,
        num_features,
        bias=True,
        fix_weight=False,
        fix_bias=False,
        inplace=False,
    ):
        """Create an ``AffineChannel`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        bias : bool, optional, default=True
            **True** to attach a bias.
        fix_weight : bool, optional, default=False
            **True** to frozen the ``weight``.
        fix_bias : bool, optional, default=False
            **True** to frozen the ``bias``.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(AffineChannel, self).__init__()
        self.num_features = num_features
        self.inplace = inplace
        if not fix_weight:
            self.weight = Parameter(init_funcs.ones(num_features))
            if inplace:
                raise ValueError('In-place operation requires fixed weight.')
        else:
            self.register_buffer('weight', init_funcs.ones(num_features))
        if bias:
            if not fix_bias:
                self.bias = Parameter(init_funcs.zeros(num_features))
            else:
                self.register_buffer('bias', init_funcs.zeros(num_features))
        else:
            self.bias = None

    def extra_repr(self):
        s = '{num_features}, ' \
            'inplace={inplace}'.format(**self.__dict__)
        if self.bias is None:
            s += ', bias=False'
        return s

    def forward(self, input):
        return array_funcs.channel_affine(
            input,
            self.weight,
            self.bias,
            dim=1,
            out=input if self.inplace else None,
        )


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
            The number of groups to split.
        num_channels : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
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
            self.register_buffer('weight', init_funcs.ones(num_channels))
            self.register_buffer('bias', init_funcs.zeros(num_channels))
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
