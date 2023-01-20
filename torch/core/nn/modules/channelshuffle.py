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
"""Channel shuffle modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional
from dragon.vm.torch.core.nn.modules.module import Module


class ChannelShuffle(Module):
    """Apply group shuffle to each channel.
    `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

    Examples:

    ```python
    m = torch.nn.ChannelShuffle(2)
    x = torch.tensor([1, 2, 3, 4])
    print(m(x))  # [1, 3, 2, 4]
    ```

    See Also
    --------
    `torch.nn.functional.channel_shuffle(...)`_

    """

    def __init__(self, groups, dim=1):
        """Create a ``ChannelShuffle`` module.

        Parameters
        ----------
        groups : int
            The number of shuffle groups.
        dim : int, optional, default=1
            The channel dimension.

        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        self.dim = dim

    def extra_repr(self):
        return 'groups={}, dim={}'.format(self.groups, self.dim)

    def forward(self, input):
        return functional.channel_shuffle(input, self.groups, self.dim)
