# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.nn import functional as F
from dragon.vm.torch.nn.modules.module import Module


class _PoolNd(Module):
    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pooling=False,
    ):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.global_pooling = global_pooling

    def extra_repr(self):
        return 'kernel_size={kernel_size}, ' \
               'stride={stride}, ' \
               'padding={padding}, ' \
               'ceil_mode={ceil_mode}, ' \
               'global_pooling={global_pooling}' \
               .format(**self.__dict__)


class AvgPool2d(_PoolNd):
    r"""Apply the 2d average pooling to input.

    The spatial output dimension is computed as:

    .. math::
        \text{Dim}_{out} = (\text{Dim}_{in} +
            2 * pad - \text{K}_{size}) / stride + 1

    Examples:

    ```python
    m = torch.nn.AvgPool2d(2, 2)
    x = torch.ones(2, 2, 4, 4)
    y = m(x)
    ```

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pooling=False,
    ):
        """Create a ``AvgPool2d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of sliding window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero-padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pooling : bool, optional
            **True** to pool globally regardless of ``kernel_size``.

        """
        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pooling=global_pooling,
        )

    def forward(self, input):
        return F.avg_pool2d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pooling=self.global_pooling,
        )


class MaxPool2d(_PoolNd):
    r"""Apply the 2d max pooling to input.

    The spatial output dimension is computed as:

    .. math::
        \text{Dim}_{out} = (\text{Dim}_{in} +
            2 * pad - \text{K}_{size}) / stride + 1

    Examples:

    ```python
    m = torch.nn.MaxPool2d(2, 2)
    x = torch.ones(2, 2, 4, 4)
    y = m(x)
    ```

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pooling=False,
    ):
        """Create a ``MaxPool2d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of sliding window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero-padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pooling : bool, optional
            **True** to pool globally regardless of ``kernel_size``.

        """
        super(MaxPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pooling=global_pooling,
        )

    def forward(self, input):
        return F.max_pool2d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pooling=self.global_pooling,
        )
