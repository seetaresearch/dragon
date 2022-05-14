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
"""Linear modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops import constant_ops
from dragon.vm.torch.core.tensor import Tensor


class Affine(Module):
    """Apply affine transformation.

    Affine is often taken as a post-processing of normalization.

    Examples:

    ```python
    m = torch.nn.Affine(5)

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
    `torch.nn.functional.affine(...)`_

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
            ``True`` to attach a bias.
        fix_weight : bool, optional, default=False
            ``True`` to freeze the ``weight``.
        fix_bias : bool, optional, default=False
            ``True`` to freeze the ``bias``.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Affine, self).__init__()
        self.num_features = num_features
        self.inplace = inplace
        if not fix_weight:
            self.weight = Parameter(constant_ops.ones(num_features))
            if inplace:
                raise ValueError('In-place operation requires fixed weight.')
        else:
            self.register_buffer('weight', constant_ops.ones(num_features))
        if bias:
            if not fix_bias:
                self.bias = Parameter(constant_ops.zeros(num_features))
            else:
                self.register_buffer('bias', constant_ops.zeros(num_features))
        else:
            self.bias = None

    def extra_repr(self):
        s = '{num_features}, ' \
            'inplace={inplace}'.format(**self.__dict__)
        if self.bias is None:
            s += ', bias=False'
        return s

    def forward(self, input):
        return F.affine(
            input,
            self.weight,
            self.bias,
            dim=1,
            out=input if self.inplace else None,
        )


class Identity(Module):
    r"""Apply the identity transformation.

    .. math:: y = x

    Examples:

    ```python
    m = torch.nn.Identity(1, unused_arg=2)
    x = torch.ones(2, 2)
    y = m(x)
    ```

    """

    def __init__(self, *args, **kwargs):
        """Create an ``Identity`` module."""
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Module):
    r"""Apply the linear transformation.

    .. math:: \text{out} = \text{input} \times \text{weight}^{T} + \text{bias}

    Examples:

    ```python
    m = torch.nn.Linear(2, 3)
    x = torch.ones(2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.linear(...)`_

    """

    def __init__(self, in_features, out_features, bias=True):
        """Create a ``Linear`` module.

        Parameters
        ----------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.

        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def extra_repr(self):
        return ('in_features={}, out_features={}, bias={}'
                .format(self.in_features, self.out_features, self.bias is not None))

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self):
        stddev = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stddev, stddev)
        if self.bias is not None:
            self.bias.data.uniform_(-stddev, stddev)
