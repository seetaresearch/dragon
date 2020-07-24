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


class RandomBBoxCrop(object):
    """Return a valid crop restricted by bounding boxes.

    Examples:

    ```python
    bbox_crop = dali.ops.RandomBBoxCrop(
        # Range of scale
        scaling=[0.3, 1.0],
        # Range of aspect ratio
        aspect_ratio=[0.5, 2.0],
        # Minimum IoUs to satisfy
        thresholds=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    )
    crop_begin, crop_size, bbox, label = bbox_crop(inputs['bbox'], inputs['label'])
    ```

    """

    def __new__(
        cls,
        scaling=(0.3, 1.0),
        aspect_ratio=(0.5, 2.0),
        thresholds=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9),
        allow_no_crop=True,
        ltrb=True,
        num_attempts=10,
    ):
        """Create a ``RandomBBoxCrop`` operator.

        Parameters
        ----------
        scaling : Sequence[float], optional, default=(0.3, 1.0)
            The range of scale for sampling regions.
        aspect_ratio : Sequence[float], optional, default=(0.5, 2.0)
            The range of aspect ratio for sampling regions.
        thresholds : Sequence[float], optional
            The minimum IoU(s) to satisfy.
        allow_no_crop : bool, optional, default=True
            **True** to include the no-cropping as a option.
        ltrb : bool, optional, default=True
            Indicate the bbox is ``ltrb`` or ``xywh`` format.
        num_attempts : int, optional, default=10
            The max number of sampling trails.

        Returns
        -------
        nvidia.dali.ops.RandomBBoxCrop
            The operator.

        """
        return ops.RandomBBoxCrop(
            scaling=scaling,
            aspect_ratio=aspect_ratio,
            thresholds=thresholds,
            allow_no_crop=allow_no_crop,
            ltrb=ltrb,
            num_attempts=num_attempts,
            device='cpu',
        )


class Slice(object):
    """Select an interval of elements from input.

    Examples:

    ```python
    slice = dali.ops.Slice(
        # Axis of intervals
        axes=[1, 0],
        # Whether the begin of interval is normalized
        # in a range of [0.0, 1.0]
        normalized_anchor=True,
        # Whether the size of interval is normalized
        # in a range of [0.0, 1.0]
        normalized_shape=True,
    )

    y = slice(inputs['x'], crop_begin, crop_size)
    ```

    """

    def __new__(
        cls,
        axes=(1, 0),
        normalized_anchor=True,
        normalized_shape=True,
    ):
        """Create a ``Slice`` operator.

        Parameters
        ----------
        axes : Sequence[int], optional
            The axis to select.
        normalized_anchor : bool, optional, default=True
            Whether the begin of interval is normalized.
        normalized_shape : bool, optional, default=True
            Whether the size of interval is normalized.

        Returns
        -------
        nvidia.dali.ops.Slice
            The operator.

        """
        return ops.Slice(
            axes=axes,
            normalized_anchor=normalized_anchor,
            device=context.get_device_type(),
        )
