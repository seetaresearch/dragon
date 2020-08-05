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
"""Padding modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.modules.utils import _ntuple
from dragon.vm.torch.core.nn.modules.utils import _pair
from dragon.vm.torch.core.nn.modules.utils import _quadruple


class _ConstantPadNd(Module):
    """The base class of constant pad."""

    def __init__(self, value):
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def extra_repr(self):
        return 'padding={}, value={}'.format(self.padding, self.value)

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)


class _ReflectionPadNd(Module):
    """The base class of reflection pad."""

    def extra_repr(self):
        return 'padding={}'.format(self.padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')


class _ReplicationPadNd(Module):
    """The base class of replication pad."""

    def extra_repr(self):
        return 'padding{}'.format(self.padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')


class ConstantPad1d(_ConstantPadNd):
    r"""Pad input according to the last dimension with a constant.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ConstantPad1d(1, 2)
    x = torch.randn(1, 2)
    y = m(x)  # (1, 2) -> (1, 4)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding, value):
        """Create a ``ConstantPad1d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 1d padding sizes.
        value : number
            The constant padding value.

        """
        super(ConstantPad1d, self).__init__(value)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    r"""Pad input according to the last 2-dimensions with a constant.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ConstantPad2d(1, 2)
    x = torch.randn(1, 2, 2)
    y = m(x)  # (1, 2, 2) -> (1, 4, 4)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding, value):
        """Create a ``ConstantPad2d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 2d padding sizes.
        value : number
            The constant padding value.

        """
        super(ConstantPad2d, self).__init__(value)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    r"""Pad input according to the last 3-dimensions with a constant.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ConstantPad3d(1, 2)
    x = torch.randn(1, 2, 2, 2)
    y = m(x)  # (1, 2, 2, 2) -> (1, 4, 4, 4)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding, value):
        """Create a ``ConstantPad3d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 3d padding sizes.
        value : number
            The constant padding value.

        """
        super(ConstantPad3d, self).__init__(value)
        self.padding = _ntuple(6)(padding)


class ReflectionPad1d(_ReflectionPadNd):
    r"""Pad input according to the last dimension by reflecting boundary.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ReflectionPad1d(1)
    x = torch.randn(1, 4)
    y = m(x)  # (1, 4) -> (1, 6)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ReflectionPad1d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 1d padding sizes.

        """
        super(ReflectionPad1d, self).__init__()
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    r"""Pad input according to the last 2-dimensions by reflecting boundary.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ReflectionPad2d(1)
    x = torch.randn(1, 4, 4)
    y = m(x)  # (1, 4, 4) -> (1, 6, 6)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ReflectionPad2d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 2d padding sizes.

        """
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReflectionPad3d(_ReflectionPadNd):
    r"""Pad input according to the last 3-dimensions by reflecting boundary.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ReflectionPad3d(1)
    x = torch.randn(1, 4, 4, 4)
    y = m(x)  # (1, 4, 4, 4) -> (1, 6, 6, 6)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ReflectionPad3d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 3d padding sizes.

        """
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)


class ReplicationPad1d(_ReplicationPadNd):
    r"""Pad input according to the last dimension by replicating boundary.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ReplicationPad1d(1)
    x = torch.randn(1, 4)
    y = m(x)  # (1, 4) -> (1, 6)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ReplicationPad1d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 1d padding sizes.

        """
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    r"""Pad input according to the last 2-dimensions by replicating boundary.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ReplicationPad2d(1)
    x = torch.randn(1, 4, 4)
    y = m(x)  # (1, 4, 4) -> (1, 6, 6)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ReplicationPad2d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 2d padding sizes.

        """
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    r"""Pad input according to the last 3-dimensions by replicating boundary.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ReplicationPad3d(1)
    x = torch.randn(1, 4, 4, 4)
    y = m(x)  # (1, 4, 4, 4) -> (1, 6, 6, 6)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ReplicationPad3d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 3d padding sizes.

        """
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)


class ZeroPad2d(ConstantPad2d):
    r"""Pad input according to the last 2-dimensions with zeros.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    Examples:

    ```python
    m = torch.nn.ZeroPad2d(1)
    x = torch.randn(1, 2, 2)
    y = m(x)  # (1, 2, 2) -> (1, 4, 4)
    ```

    See Also
    --------
    `torch.nn.functional.pad(...)`_

    """

    def __init__(self, padding):
        """Create a ``ZeroPad2d`` module.

        Parameters
        ----------
        padding : Union[int, Sequence[int]]
            The 2d padding sizes.

        """
        super(ZeroPad2d, self).__init__(padding, 0.)
