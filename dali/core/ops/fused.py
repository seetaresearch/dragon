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

from dragon.core.util import six
from dragon.vm.dali.core import context
from dragon.vm.dali.core import types


class CropMirrorNormalize(object):
    """Crop and normalize input with the horizontal flip.

    Examples:

    ```python
    flip_rng = dali.ops.CoinFlip(0.5)
    cmn = dali.ops.CropMirrorNormalize(
        # Match the number of spatial dims
        # (H, W) for 2d input
        # (D, H, W) for 3d input
        crop=(224, 224),
        # Historical values to normalize input
        mean=(102., 115., 122.),
        std=(1., 1., 1.),
        # ``BGR``, ``RGB``, or ``GRAY``
        image_type='BGR',
        # Or ``float16`` for fp16 training
        output_dtype='float32',
        # Or ``NHWC``
        output_layout='NCHW'
    )
    y = cmn(inputs['x'], mirror=flip_rng())
    ```

    """

    def __new__(
        cls,
        crop=None,
        mirror=None,
        mean=0.,
        std=1.,
        image_type='BGR',
        output_dtype='float32',
        output_layout='NCHW',
    ):
        """Create a ``CropMirrorNormalize`` operator.

        Parameters
        ----------
        crop : Sequence[int], optional
            The cropped spatial dimensions for output.
        mirror : {0, 1}, optional
            Whether to apply the horizontal flip.
        mean : Union[float, Sequence[float]], optional
            The values to subtract.
        std : Union[float, Sequence[float]], optional
            The values to divide after subtraction.
        image_type : {'BGR', 'RGB'}, optional
            The color space of input.
        output_dtype : {'float16', 'float32'}, optional
            The data type of output.
        output_layout : {'NCHW', 'NHWC'}, optional
            The data format of output.

        Returns
        -------
        nvidia.dali.ops.CropMirrorNormalize
            The operator.

        """
        if isinstance(output_dtype, six.string_types):
            output_dtype = getattr(types, output_dtype.upper())
        if isinstance(output_layout, six.string_types):
            output_layout = getattr(types, output_layout.upper())
        if isinstance(image_type, six.string_types):
            image_type = getattr(types, image_type.upper())
        return ops.CropMirrorNormalize(
            crop=crop,
            mirror=mirror,
            mean=mean,
            std=std,
            output_dtype=output_dtype,
            output_layout=output_layout,
            image_type=image_type,
            device=context.get_device_type(),
        )
