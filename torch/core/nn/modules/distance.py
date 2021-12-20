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
"""Distance modules."""

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module


class CosineSimilarity(Module):
    r"""Compute the cosine similarity.

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
        return F.cosine_similarity(x1, x2, self.dim, self.eps)
