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
"""Pooling modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module


class _PoolNd(Module):
    """Apply the n-dimension pooling."""

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.global_pool = global_pool

    def extra_repr(self):
        return 'kernel_size={kernel_size}, ' \
               'stride={stride}, ' \
               'padding={padding}, ' \
               'ceil_mode={ceil_mode}, ' \
               'global_pool={global_pool}' \
               .format(**self.__dict__)


class AvgPool1d(_PoolNd):
    r"""Apply the 1d average pooling.

    This module excepts the input size :math:`(N, C, H)`,
    and output size is :math:`(N, C, H_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of data.

    Examples:

    ```python
    m = torch.nn.AvgPool1d(2, 2)
    x = torch.ones(2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.avg_pool1d(...)`_

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        """Create a ``AvgPool1d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of pooling window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of pooling window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pool : bool, optional, default=False
            Apply the global pooling or not.

        """
        super(AvgPool1d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pool=global_pool,
        )

    def forward(self, input):
        return F.avg_pool1d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pool=self.global_pool,
        )


class AvgPool2d(_PoolNd):
    r"""Apply the 2d average pooling.

    This module excepts the input size :math:`(N, C, H, W)`,
    and output size is :math:`(N, C, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` and :math:`W` are the height and width of data.

    Examples:

    ```python
    m = torch.nn.AvgPool2d(2, 2)
    x = torch.ones(2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.avg_pool2d(...)`_

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        """Create a ``AvgPool2d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of pooling window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of pooling window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pool : bool, optional, default=False
            Apply the global pooling or not.

        """
        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pool=global_pool,
        )

    def forward(self, input):
        return F.avg_pool2d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pool=self.global_pool,
        )


class AvgPool3d(_PoolNd):
    r"""Apply the 3d average pooling.

    This module excepts the input size :math:`(N, C, D, H, W)`,
    and output size is :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`D`, :math:`H` and :math:`W` are the depth, height and width of data.

    Examples:

    ```python
    m = torch.nn.AvgPool3d(2, 2)
    x = torch.ones(2, 2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.avg_pool3d(...)`_

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        """Create a ``AvgPool3d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of pooling window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of pooling window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pool : bool, optional, default=False
            Apply the global pooling or not.

        """
        super(AvgPool3d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pool=global_pool,
        )

    def forward(self, input):
        return F.avg_pool3d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pool=self.global_pool,
        )


class MaxPool1d(_PoolNd):
    r"""Apply the 1d max pooling.

    This module excepts the input size :math:`(N, C, H)`,
    and output size is :math:`(N, C, H_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of data.

    Examples:

    ```python
    m = torch.nn.MaxPool1d(2, 2)
    x = torch.ones(2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.max_pool1d(...)`_

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        """Create a ``MaxPool1d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of pooling window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of pooling window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pool : bool, optional, default=False
            Apply the global pooling or not.

        """
        super(MaxPool1d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pool=global_pool,
        )

    def forward(self, input):
        return F.max_pool1d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pool=self.global_pool,
        )


class MaxPool2d(_PoolNd):
    r"""Apply the 2d max pooling.

    This module excepts the input size :math:`(N, C, H, W)`,
    and output size is :math:`(N, C, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` and :math:`W` are the height and width of data.

    Examples:

    ```python
    m = torch.nn.MaxPool2d(2, 2)
    x = torch.ones(2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.max_pool2d(...)`_

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        """Create a ``MaxPool2d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of pooling window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of pooling window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pool : bool, optional
            Apply the global pooling or not.

        """
        super(MaxPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pool=global_pool,
        )

    def forward(self, input):
        return F.max_pool2d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pool=self.global_pool,
        )


class MaxPool3d(_PoolNd):
    r"""Apply the 3d max pooling.

    This module excepts the input size :math:`(N, C, D, H, W)`,
    and output size is :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`D`, :math:`H` and :math:`W` are the depth, height and width of data.

    Examples:

    ```python
    m = torch.nn.MaxPool3d(2, 2)
    x = torch.ones(2, 2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.max_pool3d(...)`_

    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        global_pool=False,
    ):
        """Create a ``MaxPool3d`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of pooling window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of pooling window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        ceil_mode : bool, optional, default=False
            Ceil or floor the boundary.
        global_pool : bool, optional, default=False
            Apply the global pooling or not.

        """
        super(MaxPool3d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            global_pool=global_pool,
        )

    def forward(self, input):
        return F.max_pool3d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            global_pool=self.global_pool,
        )
