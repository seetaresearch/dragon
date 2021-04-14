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
"""Image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation
    ops = deprecation.not_installed('nvidia.dali')

from dragon.core.util import six
from dragon.vm.dali.core.framework import context
from dragon.vm.dali.core.framework import types


class Brightness(object):
    """Adjust the brightness of image.

    Examples:

    ```python
    # Historical jitter range for brightness
    twist_rng = dali.ops.Uniform(range=[0.6, 1.4])

    brightness = dali.ops.Brightness()
    y = brightness(inputs['x'], brightness=twist_rng())
    ```

    """

    def __new__(cls, **kwargs):
        """Create a ``Brightness`` operator.

        Returns
        -------
        nvidia.dali.ops.Brightness
            The operator.

        """
        return ops.Brightness(device=context.get_device_type(), **kwargs)


class BrightnessContrast(object):
    """Adjust the brightness and contrast of image.

    Examples:

    ```python
    # Historical jitter range for brightness and contrast
    twist_rng = dali.ops.Uniform(range=[0.6, 1.4])

    bc = dali.ops.BrightnessContrast()
    y = bc(inputs['x'], brightness=twist_rng(), contrast=twist_rng())
    ```

    """

    def __new__(cls, **kwargs):
        """Create a ``BrightnessContrast`` operator.

        Returns
        -------
        nvidia.dali.ops.BrightnessContrast
            The operator.

        """
        return ops.BrightnessContrast(device=context.get_device_type(), **kwargs)


class Contrast(object):
    """Adjust the contrast of image.

    Examples:

    ```python
    # Historical jitter range for contrast
    twist_rng = dali.ops.Uniform(range=[0.6, 1.4])

    contrast = dali.ops.Contrast()
    y = contrast(inputs['x'], contrast=twist_rng())
    ```

    """

    def __new__(cls, **kwargs):
        """Create a ``Contrast`` operator.

        Returns
        -------
        nvidia.dali.ops.Contrast
            The operator.

        """
        return ops.Contrast(device=context.get_device_type(), **kwargs)


class CropMirrorNormalize(object):
    """Crop and normalize image with the horizontal flip.

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
        # Or ``float16`` for fp16 training
        dtype='float32',
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
        dtype='float32',
        output_layout='NCHW',
        **kwargs
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
        dtype : {'float16', 'float32'}, optional
            The data type of output.
        output_layout : {'NCHW', 'NHWC'}, optional
            The data format of output.

        Returns
        -------
        nvidia.dali.ops.CropMirrorNormalize
            The operator.

        """
        if isinstance(dtype, six.string_types):
            dtype = getattr(types, dtype.upper())
        if isinstance(output_layout, six.string_types):
            output_layout = getattr(types, output_layout.upper())
        return ops.CropMirrorNormalize(
            crop=crop,
            mirror=mirror,
            mean=mean,
            std=std,
            dtype=dtype,
            output_layout=output_layout,
            device=context.get_device_type(),
            **kwargs
        )


class Hsv(object):
    """Adjust the hue and saturation.

    Examples:

    ```python
    # Historical jitter range for saturation
    twist_rng = dali.ops.Uniform(range=[0.6, 1.4])

    hsv = dali.ops.Hsv()
    y = hsv(inputs['x'], saturation=twist_rng())
    ```

    """

    def __new__(cls, **kwargs):
        """Create a ``Hsv`` operator.

        Returns
        -------
        nvidia.dali.ops.Hsv
            The operator.

        """
        return ops.Hsv(device=context.get_device_type(), **kwargs)


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
        **kwargs
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
            **kwargs
        )


class RandomBBoxCrop(object):
    """Return an valid image crop restricted by bounding boxes.

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
        num_attempts=10,
        bbox_layout='xyXY',
        **kwargs
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
            ``True`` to include the no-cropping as an option.
        num_attempts : int, optional, default=10
            The max number of sampling trails.
        bbox_layout : str, optional, default='xyXY'
            The optional bbox layout.

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
            num_attempts=num_attempts,
            bbox_layout=bbox_layout,
            device='cpu',
            **kwargs
        )


class RandomResizedCrop(object):
    """Return a resized random crop of image.

    Examples:

    ```python
    resize = dali.ops.RandomResizedCrop(
        size=(224, 224),
        # Inception sampling policy for image classification
        random_area=(0.08, 1.00),
        random_aspect_ratio=(0.75, 1.33),
    )
    y = resize(inputs['x'])
    ```

    """

    def __new__(
        cls,
        size,
        interp_type=None,
        mag_filter=None,
        min_filter=None,
        random_area=(0.08, 1.),
        random_aspect_ratio=(0.75, 1.33),
        num_attempts=10,
        **kwargs
    ):
        """Create a ``RandomResizedCrop`` operator.

        Parameters
        ----------
        size : Union[int, Sequence[int]]
            The output image size.
        interp_type : str, optional
            The interpolation for both up and down sampling.
        mag_filter : str, optional, default='LINEAR'
            The interpolation for up sampling.
        min_filter : str, optional, default='TRIANGULAR'
            The interpolation for down sampling.
        random_area : Sequence[float], optional, default=(0.08, 1.)
            The range of scale for sampling.
        random_aspect_ratio : Sequence[float], optional, default=(0.75, 1.33)
            The range of aspect ratio for sampling.
        num_attempts : int, optional, default=10
            The max number of sampling trails.

        Returns
        -------
        nvidia.dali.ops.RandomResizedCrop
            The operator.

        """
        if isinstance(interp_type, six.string_types):
            interp_type = getattr(types, 'INTERP_' + interp_type.upper())
        if isinstance(mag_filter, six.string_types):
            mag_filter = getattr(types, 'INTERP_' + mag_filter.upper())
        if isinstance(min_filter, six.string_types):
            min_filter = getattr(types, 'INTERP_' + min_filter.upper())
        return ops.RandomResizedCrop(
            size=size,
            interp_type=interp_type,
            mag_filter=mag_filter,
            min_filter=min_filter,
            random_area=random_area,
            random_aspect_ratio=random_aspect_ratio,
            num_attempts=num_attempts,
            device=context.get_device_type(),
            **kwargs
        )


class Resize(object):
    """Resize the image.

    Examples:

    ```python
    # Resize to a fixed area
    resize1 = dali.ops.Resize(resize_x=300, resize_y=300)

    # Resize along the shorter side
    resize2 = dali.ops.Resize(resize_shorter=600, max_size=1000)

    # Resize along the longer side
    resize3 = dali.ops.Resize(resize_longer=512)
    ```

    """

    def __new__(
        cls,
        resize_x=None,
        resize_y=None,
        resize_shorter=None,
        resize_longer=None,
        max_size=None,
        interp_type=None,
        mag_filter=None,
        min_filter=None,
        **kwargs
    ):
        """Create a ``Resize`` operator.

        Parameters
        ----------
        resize_x : int, optional
            The output image width.
        resize_y : int, optional
            The output image height.
        resize_shorter : int, optional
            Resize along the shorter side and limited by ``max_size``.
        resize_longer : int, optional
            Resize along the longer side.
        max_size : int, optional, default=0
            The limited size for ``resize_shorter``.
        interp_type : str, optional
            The interpolation for both up and down sampling.
        mag_filter : str, optional, default='LINEAR'
            The interpolation for up sampling.
        min_filter : str, optional, default='TRIANGULAR'
            The interpolation for down sampling.

        Returns
        ------
        nvidia.dali.ops.Resize
            The operator.

        """
        if isinstance(interp_type, six.string_types):
            interp_type = getattr(types, 'INTERP_' + interp_type.upper())
        if isinstance(mag_filter, six.string_types):
            mag_filter = getattr(types, 'INTERP_' + mag_filter.upper())
        if isinstance(min_filter, six.string_types):
            min_filter = getattr(types, 'INTERP_' + min_filter.upper())
        return ops.Resize(
            resize_x=resize_x,
            resize_y=resize_y,
            resize_shorter=resize_shorter,
            resize_longer=resize_longer,
            max_size=max_size,
            interp_type=interp_type,
            mag_filter=mag_filter,
            min_filter=min_filter,
            device=context.get_device_type(),
            **kwargs
        )
