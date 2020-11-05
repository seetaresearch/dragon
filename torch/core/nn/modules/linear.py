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


class Linear(Module):
    r"""Apply the linear transformation.

    .. math:: y = Wx + b

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
            **True** to add a bias on the output.

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
        """
        Return a string representation of this feature.

        Args:
            self: (todo): write your description
        """
        return ('in_features={}, out_features={}, bias={}'
                .format(self.in_features, self.out_features, self.bias is not None))

    def forward(self, input):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self):
        """
        Reset the parameters.

        Args:
            self: (todo): write your description
        """
        stddev = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stddev, stddev)
        if self.bias is not None:
            self.bias.data.uniform_(-stddev, stddev)
