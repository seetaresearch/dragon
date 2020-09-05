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
"""BBox ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation
    ops = deprecation.not_installed('nvidia.dali')

from dragon.vm.dali.core.framework import context


class BbFlip(object):
    """Flip the bounding boxes.

    Examples:

    ```python
    bbox_flip = dali.ops.BbFlip()
    flip_rng = dali.ops.CoinFlip(0.5)
    bbox = bbox_flip(inputs['bbox'], horizontal=flip_rng())
    ```

    """

    def __new__(cls, horizontal=None, vertical=None, ltrb=True, **kwargs):
        """Create a ``BbFlip`` operator.

        Parameters
        ----------
        horizontal : int, optional
            Whether to apply the horizontal flip.
        vertical : int, optional
            Whether to apply the vertical flip.
        ltrb : bool, optional, default=True
            Indicate the bbox is ``ltrb`` or ``xywh`` format.

        Returns
        -------
        nvidia.dali.ops.BbFlip
            The operator.

        """
        return ops.BbFlip(
            horizontal=horizontal,
            vertical=vertical,
            ltrb=ltrb,
            device=context.get_device_type(),
            **kwargs
        )


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
    ```

    """

    def __new__(
        cls,
        ltrb=True,
        ratio=None,
        paste_x=None,
        paste_y=None,
        **kwargs
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
            **kwargs
        )
