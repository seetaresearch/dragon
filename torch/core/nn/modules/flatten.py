# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Flatten modules."""

from dragon.vm.torch.core.nn.modules.module import Module


class Flatten(Module):
    """Flatten input dimensions.

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
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
