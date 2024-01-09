# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Pixel shuffle modules."""

from dragon.vm.torch.core.nn import functional
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
        inplace_str = ", inplace" if self.inplace else ""
        return "upscale_factor={}{}".format(self.upscale_factor, inplace_str)

    def forward(self, input):
        return functional.pixel_shuffle(input, self.upscale_factor)


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
        inplace_str = ", inplace" if self.inplace else ""
        return "downscale_factor={}{}".format(self.downscale_factor, inplace_str)

    def forward(self, input):
        return functional.pixel_unshuffle(input, self.downscale_factor)
