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
"""Distance modules."""

from dragon.vm.torch.core.nn import functional
from dragon.vm.torch.core.nn.modules.module import Module


class CosineSimilarity(Module):
    r"""Compute cosine similarity.

    The ``CosineSimilarity`` function is defined as:

    .. math:: \text{CosineSimilarity}(x1, x2) =
        \frac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Examples:

    ```python
    m = torch.nn.CosineSimilarity()
    x1 = torch.randn(10, 10)
    x2 = torch.randn(10, 10)
    distance = m(x1, x2)
    ```

    See Also
    --------
    `torch.nn.functional.cosine_similarity(...)`_

    """

    def __init__(self, dim=1, eps=1e-8):
        """Create ``CosineSimilarity`` module.

        Parameters
        ----------
        dim : int, optional, default=1
            The vector dimension.
        eps : float, optional, default=1e-8
            The epsilon value.

        """
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return functional.cosine_similarity(x1, x2, self.dim, self.eps)
