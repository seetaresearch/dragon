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
"""Upsampling modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module


class Upsample(Module):
    """Upsample input via interpolating neighborhoods.

    Set :attr:`size` or :attr:`scale_factor` to determine the output size:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = torch.nn.Upsample(size=6)(x)  # Size: (1, 2, 6, 6)
    z = torch.nn.UpSample(scale_factor=2)(x)  # Size: (1, 2, 6, 8)
    ```

    The interpolating method can be set by :attr:`mode`:

    ```python
    upsample_nn = torch.nn.Upsample(mode='nearest')
    upsample_linear = torch.nn.UpSample(mode='linear')
    ```

    See Also
    --------
    `torch.nn.functional.interpolate(...)`_

    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=False,
    ):
        """Create an ``Upsample`` module.

        Parameters
        ----------
        size : Union[int, Sequence[int]], optional
            The output size.
        scale_factor : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.
        mode : str, optional, default='nearest'
            ``'nearest'`` or ``'linear'``.
        align_corners : bool, optional, default=False
            Whether to align corners in ``'linear'`` interpolating.

        """
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, (tuple, list)):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class UpsamplingBilinear2d(Upsample):
    """Upsample input via bilinear interpolating.

    Set :attr:`size` or :attr:`scale_factor` to determine the output size:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = torch.nn.UpsamplingBilinear2d(size=6)(x)  # Size: (1, 2, 6, 6)
    z = torch.nn.UpsamplingBilinear2d(scale_factor=2)(x)  # Size: (1, 2, 6, 8)
    ```

    See Also
    --------
    `torch.nn.functional.interpolate(...)`_

    """

    def __init__(self, size=None, scale_factor=None):
        """Create an ``UpsamplingBilinear2d`` module.

        Parameters
        ----------
        size : Union[int, Sequence[int]], optional
            The output size.
        scale_factor : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.

        """
        super(UpsamplingBilinear2d, self).__init__(
            size, scale_factor, mode='linear', align_corners=True)


class UpsamplingNearest2d(Upsample):
    """Upsample input via nearest interpolating.

    Set :attr:`size` or :attr:`scale_factor` to determine the output size:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = torch.nn.UpsamplingNearest2d(size=6)(x)  # Size: (1, 2, 6, 6)
    z = torch.nn.UpsamplingNearest2d(scale_factor=2)(x)  # Size: (1, 2, 6, 8)
    ```

    See Also
    --------
    `torch.nn.functional.interpolate(...)`_

    """

    def __init__(self, size=None, scale_factor=None):
        """Create an ``UpsamplingNearest2d`` module.

        Parameters
        ----------
        size : Union[int, Sequence[int]], optional
            The output size.
        scale_factor : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.

        """
        super(UpsamplingNearest2d, self).__init__(
            size, scale_factor, mode='nearest')
