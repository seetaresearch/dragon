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
"""Fold modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module


class Unfold(Module):
    r"""Extract the sliding blocks.

    This module excepts the input size :math:`(N, C, D1, D2, ...)`,
    and output size is :math:`(N, C \times \prod(\text{kernel\_size}), L)`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`L` is :math:`\prod(D_{\text{out}})`.

    Examples:

    ```python
    m = torch.nn.Unfold(3, padding=1)
    x = torch.ones(2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.unfold(...)`_

    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        """Create a ``Unfold`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of sliding window.
        dilation : Union[int, Sequence[int]], optional, default=1
            The dilated rate of sliding convolution.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.

        """
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def extra_repr(self):
        return 'kernel_size={kernel_size}, ' \
               'dilation={dilation}, ' \
               'padding={padding}, ' \
               'stride={stride}' \
               .format(**self.__dict__)

    def forward(self, input):
        return F.unfold(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)
