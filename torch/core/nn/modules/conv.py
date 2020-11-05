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
"""Convolution modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.modules.utils import _pair
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.tensor import Tensor


class _ConvNd(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
    ):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            padding: (str): write your description
            dilation: (todo): write your description
            transposed: (bool): write your description
            output_padding: (str): write your description
            groups: (list): write your description
            bias: (float): write your description
        """
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the kernel parameters.

        Args:
            self: (todo): write your description
        """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        """
        Return a string representation of - repr.

        Args:
            self: (todo): write your description
        """
        s = ('{in_channels}, '
             '{out_channels}, '
             'kernel_size={kernel_size}, '
             'stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2d(_ConvNd):
    """Apply the 2d convolution.

    Examples:

    ```python
    m = torch.nn.Conv2d(2, 3, 3, padding=1)
    x = torch.ones(2, 2, 4, 4)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.conv2d(...)`_

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """Create a ``Conv2d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution kernel.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero-padding size.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution kernel.
        groups : int, optional, default=1
            The number of groups to split input channels.
        bias : bool, optional, default=True
            **True** to add a bias on the output.

        """
        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return F.conv2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose2d(_ConvNd):
    """Apply the 2d deconvolution.

    Examples:

    ```python
    m = torch.nn.ConvTranspose2d(2, 3, 2, stride=2)
    x = torch.ones(2, 2, 4, 4)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.conv_transpose2d(...)`_

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ):
        """Create a ``ConvTranspose2d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution kernel.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero-padding size.
        output_padding : int, optional, default=1
            The additional padding size.
        groups : int, optional, default=1
            The number of groups to split input channels.
        bias : bool, optional, default=True
            **True** to add a bias on the output.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution kernel.

        """
        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            transposed=True,
            output_padding=_pair(output_padding),
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return F.conv_transpose2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class DepthwiseConv2d(Conv2d):
    """Apply the 2d depthwise convolution.

    Examples:

    ```python
    m = torch.nn.DepthwiseConv2d(3, 3, 3, padding=1)
    x = torch.ones(2, 3, 4, 4)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.depthwise_conv2d(...)`_

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        """Create a ``DepthwiseConv2d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution kernel.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero-padding size.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated kernel.
        bias : bool, optional, default=True
            **True** to add a bias on the output.

        """
        super(DepthwiseConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            groups=in_channels,
            bias=bias,
        )

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return F.depthwise_conv2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
