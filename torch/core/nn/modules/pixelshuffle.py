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
"""Pixel shuffle modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module


class PixelShuffle(Module):
    """Rearrange depth elements into pixels.

    Examples:

    ```python
    m = torch.nn.PixelShuffle(2)
    x = torch.arange(1 * 12 * 1 * 1).reshape((1, 12, 1, 1))
    print(m(x).size())  # [1, 3, 2, 2]
    ```

    See Also
    --------
    `torch.nn.functional.pixel_shuffle(...)`_

    """

    def __init__(self, upscale_factor, inplace=False):
        """Create a ``PixelShuffle`` module.

        Parameters
        ----------
        upscale_factor : int
            The factor to upscale pixels.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'upscale_factor={}{}'.format(self.upscale_factor, inplace_str)

    def forward(self, input):
        return F.pixel_shuffle(input, self.upscale_factor)


class PixelUnshuffle(Module):
    """Rearrange pixels into depth elements.

    Examples:

    ```python
    m = torch.nn.PixelUnshuffle(2)
    x = torch.arange(1 * 3 * 2 * 2).reshape((1, 3, 2, 2))
    print(m(x).size())  # [1, 12, 1, 1]
    ```

    See Also
    --------
    `torch.nn.functional.pixel_unshuffle(...)`_

    """

    def __init__(self, downscale_factor, inplace=False):
        """Create a ``PixelUnshuffle`` module.

        Parameters
        ----------
        downscale_factor : int
            The factor to downscale pixels.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'downscale_factor={}{}'.format(self.downscale_factor, inplace_str)

    def forward(self, input):
        return F.pixel_unshuffle(input, self. downscale_factor)
