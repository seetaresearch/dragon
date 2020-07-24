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


class BbFlip(object):
    """Flip the bounding boxes.

    Examples:

    ```python
    bbox_flip = dali.ops.BbFlip()
    flip_rng = dali.ops.CoinFlip(0.5)
    bbox = bbox_flip(inputs['bbox'], horizontal=flip_rng())
    ```

    """

    def __new__(cls, horizontal=None, vertical=None, ltrb=True):
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
        )
