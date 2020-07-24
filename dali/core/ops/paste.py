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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import ops
except ImportError:
    ops = None

from dragon.vm.dali.core import context


class BBoxPaste(object):
    """Transform bounding boxes to match the ``Paste`` operator.

    Examples:

    ```python
    bbox_paste = dali.ops.BBoxPaste()
    paste_pos = dali.ops.Uniform((0., 1.))
    paste_ratio = dali.ops.Uniform((0., 3.))
    paste_prob = dali.ops.CoinFlip(0.5)

    bbox = bbox_paste(
        inputs['bbox'],
        # Expand ratio
        ratio=paste_ratio() * paste_prob() + 1.,
        # PosX, PosY
        paste_x=paste_pos(),
        paste_y=paste_pos(),
    )
    """

    def __new__(
        cls,
        ltrb=True,
        ratio=None,
        paste_x=None,
        paste_y=None,
    ):
        """Create a ``BBoxPaste`` operator.

        Parameters
        ----------
        ltrb : bool, optional, default=True
            Indicate the bbox is ``ltrb`` or ``xywh`` format.
        ratio : int, optional
            The expand ratio.
        paste_x : int, optional
            The paste position at x-axis.
        paste_y : int, optional
            The paste position at y-axis.

        Returns
        -------
        nvidia.dali.ops.BBoxPaste
            The operator.

        """
        return ops.BBoxPaste(
            ltrb=ltrb,
            ratio=ratio,
            paste_x=paste_x,
            paste_y=paste_y,
            device='cpu',
        )


class Paste(object):
    """Copy image into a larger canvas.

    Examples:

    ```python
    paste = dali.ops.Paste(
        # The image channels
        n_channels=3,
        # Historical values before mean subtraction
        fill_value=(102., 115., 122.),
    )
    paste_pos = dali.ops.Uniform((0., 1.))
    paste_ratio = dali.ops.Uniform((0., 3.))
    paste_prob = dali.ops.CoinFlip(0.5)

    y = paste(
        inputs['x'],
        # Expand ratio
        ratio=paste_ratio() * paste_prob() + 1.,
        # PosX, PosY
        paste_x=paste_pos(),
        paste_y=paste_pos(),
    )
    ```

    """

    def __new__(
        cls,
        n_channels=3,
        fill_value=(0., 0., 0.),
        ratio=None,
        paste_x=None,
        paste_y=None,
    ):
        """Create a ``Paste`` operator.

        Parameters
        ----------
        n_channels : int, optional, default=3
            The image channels.
        fill_value : Sequence[number], optional
            The value(s) to fill for the canvas.
        ratio : int, optional
            The expand ratio.
        paste_x : int, optional
            The paste position at x-axis.
        paste_y : int, optional
            The paste position at y-axis.

        Returns
        -------
        nvidia.dali.ops.Paste
            The operator.

        """
        return ops.Paste(
            n_channels=n_channels,
            fill_value=fill_value,
            ratio=ratio,
            paste_x=paste_x,
            paste_y=paste_y,
            device=context.get_device_type(),
        )
