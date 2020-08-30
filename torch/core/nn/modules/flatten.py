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
"""Flatten modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn.modules.module import Module


class Flatten(Module):
    """Flatten the dimensions of input.

    Examples:

    ```python
    m = torch.nn.Flatten()
    x = torch.ones(1, 2, 4, 4)
    y = m(x)
    print(y.size())  # (1, 32)
    ```

    See Also
    --------
    `torch.flatten(...)`_

    """

    def __init__(self, start_dim=1, end_dim=-1):
        """Create a ``Flatten`` module.

        Parameters
        ----------
        start_dim : int, optional, default=0
            The start dimension to flatten.
        end_dim : int, optional, default=-1
            The end dimension to flatten.

        """
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def extra_repr(self):
        return 'start_dim={}, end_dim={}' \
               .format(self.start_dim, self.end_dim)

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
