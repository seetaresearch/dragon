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
"""Dropout modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module


class DropBlock2d(Module):
    r"""Set the blocks to zero randomly.
    `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

    The **DropBlock** function is defined as:

    .. math::
        \text{DropBlock}(x_{ijk}) =
            x_{ijk} * (r_{ik} \sim \mathcal{B}(1, 1 - \gamma)) \\ \quad \\
        \text{where}\quad \gamma =
            \frac{p}{\text{block\_size}^{n}}
            \frac{\text{feat\_size}^{n}}{(\text{feat\_size} - \text{block\_size} + 1)^n}

    Examples:

    ```python
    x = torch.ones(1, 3, 4, 4)
    m = torch.nn.DropBlock2d(block_size=3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.drop_block2d(...)`_

    """

    def __init__(self, p=0.1, block_size=7, inplace=False):
        r"""Create a ``DropBlock2d`` module.

        Parameters
        ----------
        p : float, optional, default=0.1
            The dropping ratio.
        block_size : int, optional, default=7
            The size of a spatial block.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(DropBlock2d, self).__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}, block_size={}{}' \
               .format(self.p, self.block_size, inplace_str)

    def forward(self, input):
        return F.drop_block2d(
            input, self.p,
            block_size=self.block_size,
            training=self.training,
            inplace=self.inplace,
        )


class Dropout(Module):
    r"""Set the elements to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * (r \sim \mathcal{B}(1, 1 - p))

    Examples:

    ```python
    # Dropout is enabled if the module is at ``training``
    m = torch.nn.Dropout(p=0.5, inplace=True)
    x = torch.ones(2, 3)
    y = m(x)

    # Nothing will happen if it set to ``evaluating``
    m.eval()
    z = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.dropout(...)`_

    """

    def __init__(self, p=0.5, inplace=False):
        """Create a ``Dropout`` module.

        Parameters
        ----------
        p : float, optional, default=0.5
            The dropping ratio.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)


class DropPath(Module):
    r"""Set the examples over input to zero randomly.
    `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

    The **DropPath** function is defined as:

    .. math:: \text{DropPath}(x_{ij}) = x_{ij} * (r_{i} \sim \mathcal{B}(1, 1 - p))

    Examples:

    ```python
    x = torch.ones(5, 2, 2, 2)
    m = torch.nn.DropPath()
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.drop_path(...)`_

    """

    def __init__(self, p=0.2, inplace=False):
        """Create a ``DropPath`` module.

        Parameters
        ----------
        p : float, optional, default=0.2
            The dropping ratio.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(DropPath, self).__init__()
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)

    def forward(self, input):
        return F.drop_path(input, self.p, self.training, self.inplace)
