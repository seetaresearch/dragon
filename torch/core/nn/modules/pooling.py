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


class _AdaptivePoolNd(Module):
    """Apply the n-dimension adaptive pooling."""

    def __init__(self, output_size):
        super(_AdaptivePoolNd, self).__init__()
        self.output_size = output_size

    def extra_repr(self):
        return 'output_size={}'.format(self.output_size)


class _PoolNd(Module):
    """Apply the n-dimension pooling."""

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
    ):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, ' \
               'stride={stride}, ' \
               'padding={padding}, ' \
               'ceil_mode={ceil_mode}' \
               .format(**self.__dict__)


class AdaptiveAvgPool1d(_AdaptivePoolNd):
    r"""Apply the 1d adaptive average pooling.

    This module excepts the input size :math:`(N, C, H)`,
    and output size is :math:`(N, C, H_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of data.

    Examples:

    ```python
    m = torch.nn.AdaptiveAvgPool1d(1)
    x = torch.ones(2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.adaptive_avg_pool1d(...)`_

    """

    def __init__(self, output_size):
        """Create a ``AdaptiveAvgPool1d`` module.

        Parameters
        ----------
        output_size : Union[int, Sequence[int]]
            The target output size.

        """
        super(AdaptiveAvgPool1d, self).__init__(output_size=output_size)

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool2d(_AdaptivePoolNd):
    r"""Apply the 2d adaptive average pooling.

    This module excepts the input size :math:`(N, C, H, W)`,
    and output size is :math:`(N, C, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` and :math:`W` are the height and width of data.

    Examples:

    ```python
    m = torch.nn.AdaptiveAvgPool2d(1)
    x = torch.ones(2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.adaptive_avg_pool2d(...)`_

    """

    def __init__(self, output_size):
        """Create a ``AdaptiveAvgPool2d`` module.

        Parameters
        ----------
        output_size : Union[int, Sequence[int]]
            The target output size.

        """
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)


class AdaptiveAvgPool3d(_AdaptivePoolNd):
    r"""Apply the 3d adaptive average pooling.

    This module excepts the input size :math:`(N, C, D, H, W)`,
    and output size is :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`D`, :math:`H` and :math:`W` are the depth, height and width of data.

    Examples:

    ```python
    m = torch.nn.AdaptiveAvgPool3d(1)
    x = torch.ones(2, 2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.adaptive_avg_pool3d(...)`_

    """

    def __init__(self, output_size):
        """Create a ``AdaptiveAvgPool3d`` module.

        Parameters
        ----------
        output_size : Union[int, Sequence[int]]
            The target output size.

        """
        super(AdaptiveAvgPool3d, self).__init__(output_size=output_size)

    def forward(self, input):
        return F.adaptive_avg_pool3d(input, self.output_size)


class AdaptiveMaxPool1d(_AdaptivePoolNd):
    r"""Apply the 1d adaptive max pooling.

    This module excepts the input size :math:`(N, C, H)`,
    and output size is :math:`(N, C, H_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of data.

    Examples:

    ```python
    m = torch.nn.AdaptiveMaxPool1d(1)
    x = torch.ones(2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.adaptive_max_pool1d(...)`_

    """

    def __init__(self, output_size):
        """Create a ``AdaptiveMaxPool1d`` module.

        Parameters
        ----------
        output_size : Union[int, Sequence[int]]
            The target output size.

        """
        super(AdaptiveMaxPool1d, self).__init__(output_size=output_size)

    def forward(self, input):
        return F.adaptive_max_pool1d(input, self.output_size)


class AdaptiveMaxPool2d(_AdaptivePoolNd):
    r"""Apply the 2d adaptive max pooling.

    This module excepts the input size :math:`(N, C, H, W)`,
    and output size is :math:`(N, C, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` and :math:`W` are the height and width of data.

    Examples:

    ```python
    m = torch.nn.AdaptiveMaxPool2d(1)
    x = torch.ones(2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.adaptive_max_pool2d(...)`_

    """

    def __init__(self, output_size):
        """Create a ``AdaptiveMaxPool2d`` module.

        Parameters
        ----------
        output_size : Union[int, Sequence[int]]
            The target output size.

        """
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size)

    def forward(self, input):
        return F.adaptive_max_pool2d(input, self.output_size)


class AdaptiveMaxPool3d(_AdaptivePoolNd):
    r"""Apply the 3d adaptive max pooling.

    This module excepts the input size :math:`(N, C, D, H, W)`,
    and output size is :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`D`, :math:`H` and :math:`W` are the depth, height and width of data.

    Examples:

    ```python
    m = torch.nn.AdaptiveMaxPool3d(1)
    x = torch.ones(2, 2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.adaptive_max_pool3d(...)`_

    """

    def __init__(self, output_size):
        """Create a ``AdaptiveMaxPool3d`` module.

        Parameters
        ----------
        output_size : Union[int, Sequence[int]]
            The target output size.

        """
        super(AdaptiveMaxPool3d, self).__init__(output_size=output_size)

    def forward(self, input):
        return F.adaptive_max_pool3d(input, self.output_size)


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

        """
        super(AvgPool1d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        return F.avg_pool1d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
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

        """
        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        return F.avg_pool2d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
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

        """
        super(AvgPool3d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        return F.avg_pool3d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
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

        """
        super(MaxPool1d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        return F.max_pool1d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
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

        """
        super(MaxPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        return F.max_pool2d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
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

        """
        super(MaxPool3d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        return F.max_pool3d(
            input=input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )
