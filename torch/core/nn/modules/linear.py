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
from dragon.vm.torch.core.tensor import Tensor


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
