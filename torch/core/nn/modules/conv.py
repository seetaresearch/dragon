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
from dragon.vm.torch.core.nn.modules.utils import _single
from dragon.vm.torch.core.nn.modules.utils import _triple
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.tensor import Tensor


class _ConvNd(Module):
    """Apply the n-dimension convolution."""

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
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('<in_channels> must be divisible by <groups>.')
        if out_channels % groups != 0:
            raise ValueError('<out_channels> must be divisible by <groups>.')
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
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stddev = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stddev, stddev)
        if self.bias is not None:
            self.bias.data.uniform_(-stddev, stddev)

    def extra_repr(self):
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


class Conv1d(_ConvNd):
    r"""Apply the 1d convolution.

    This module excepts the input size :math:`(N, C_{\text{in}}, H)`,
    and output size is :math:`(N, C_{\text{out}}, H_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of data.

    Examples:

    ```python
    m = torch.nn.Conv1d(2, 3, 3, padding=1)
    x = torch.ones(2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.conv1d(...)`_

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
        """Create a ``Conv1d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        groups : int, optional, default=1
            The number of groups to split channels into.
        bias : bool, optional, default=True
            ``True`` to add a bias on the output.

        """
        super(Conv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        return F.conv1d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2d(_ConvNd):
    r"""Apply the 2d convolution.

    This module excepts the input size :math:`(N, C_{\text{in}}, H, W)`,
    and output size is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`H` and :math:`W` are the height and width of data.

    Examples:

    ```python
    m = torch.nn.Conv2d(2, 3, 3, padding=1)
    x = torch.ones(2, 2, 2, 2)
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
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        groups : int, optional, default=1
            The number of groups to split channels into.
        bias : bool, optional, default=True
            ``True`` to add a bias on the output.

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
        return F.conv2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv3d(_ConvNd):
    r"""Apply the 3d convolution.

    This module excepts the input size :math:`(N, C_{\text{in}}, D, H, W)`,
    and output size is :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`D`, :math:`H` and :math:`W` are the depth, height and width of data.

    Examples:

    ```python
    m = torch.nn.Conv3d(2, 3, 3, padding=1)
    x = torch.ones(2, 2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.conv3d(...)`_

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
        """Create a ``Conv3d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        groups : int, optional, default=1
            The number of groups to split channels into.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.

        """
        super(Conv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            transposed=False,
            output_padding=_triple(0),
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        return F.conv3d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose1d(_ConvNd):
    """Apply the 1d deconvolution.

    Examples:

    ```python
    m = torch.nn.ConvTranspose2d(2, 3, 2, stride=2)
    x = torch.ones(2, 2, 1)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.conv_transpose1d(...)`_

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
        """Create a ``ConvTranspose1d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        output_padding : int, optional, default=1
            The additional size added to the output shape.
        groups : int, optional, default=1
            The number of groups to split channels into.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.

        """
        super(ConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            transposed=True,
            output_padding=_single(output_padding),
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        return F.conv_transpose1d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose2d(_ConvNd):
    """Apply the 2d deconvolution.

    Examples:

    ```python
    m = torch.nn.ConvTranspose2d(2, 3, 2, stride=2)
    x = torch.ones(2, 2, 1, 1)
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
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        output_padding : int, optional, default=1
            The additional size added to the output shape.
        groups : int, optional, default=1
            The number of groups to split channels into.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.

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


class ConvTranspose3d(_ConvNd):
    """Apply the 3d deconvolution.

    Examples:

    ```python
    m = torch.nn.ConvTranspose3d(2, 3, 2, stride=2)
    x = torch.ones(2, 2, 1, 1, 1)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.conv_transpose3d(...)`_

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
        """Create a ``ConvTranspose3d`` module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : Union[int, Sequence[int]]
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        output_padding : int, optional, default=1
            The additional size added to the output shape.
        groups : int, optional, default=1
            The number of groups to split channels into.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.

        """
        super(ConvTranspose3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            transposed=True,
            output_padding=_triple(output_padding),
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        return F.conv_transpose3d(
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
            The size of convolution window.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.
        dilation : Union[int, Sequence[int]], optional, default=1
            The rate of dilated kernel.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.

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
        return F.depthwise_conv2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
