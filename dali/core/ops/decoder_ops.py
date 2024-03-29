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
"""Decoder ops."""

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation

    ops = deprecation.not_installed("nvidia.dali")

from dragon.vm.dali.core.framework import context
from dragon.vm.dali.core.framework import types


class ImageDecoder(object):
    """Decode image from bytes.

    Examples:

    ```python
    decode = dali.ops.ImageDecoder()
    y = decode(inputs['x'], out_type='BGR')
    ```

    """

    def __new__(
        cls,
        output_type="BGR",
        host_memory_padding=8388608,
        device_memory_padding=16777216,
        **kwargs
    ):
        """Create a ``ImageDecoder`` operator.

        Parameters
        ----------
        output_type : str, optional, default='BGR'
            The output color space.
        host_memory_padding : int, optional, default=8388608
            The number of bytes for host buffer.
        device_memory_padding : int, optional, default=16777216
            The number of bytes for device buffer.

        Returns
        -------
        nvidia.dali.ops.decoders.Image
            The operator.

        """
        if isinstance(output_type, str):
            output_type = getattr(types, output_type)
        return ops.decoders.Image(
            output_type=output_type,
            host_memory_padding=host_memory_padding,
            device_memory_padding=device_memory_padding,
            device=context.get_device_type(mixed=True),
            **kwargs
        )


class ImageDecoderRandomCrop(object):
    """Decode image and return a random crop.

    Examples:

    ```python
    decode = dali.ops.ImageDecoderRandomCrop(
        out_type='BGR',
        # Inception sampling policy for image classification
        random_area=(0.08, 1.00),
        random_aspect_ratio=(0.75, 1.33),
    )
    y = decode(inputs['x'])
    ```

    """

    def __new__(
        cls,
        output_type="BGR",
        host_memory_padding=8388608,
        device_memory_padding=16777216,
        random_area=(0.08, 1.0),
        random_aspect_ratio=(0.75, 1.33),
        num_attempts=10,
        **kwargs
    ):
        """Create a ``ImageDecoderRandomCrop`` operator.

        Parameters
        ----------
        output_type : str, optional, default='BGR'
            The output color space.
        host_memory_padding : int, optional, default=8388608
            The number of bytes for host buffer.
        device_memory_padding : int, optional, default=16777216
            The number of bytes for device buffer.
        random_area : Sequence[float], optional, default=(0.08, 1.)
            The range of scale for sampling.
        random_aspect_ratio : Sequence[float], optional, default=(0.75, 1.33)
            The range of aspect ratio for sampling.
        num_attempts : int, optional, default=10
            The max number of sampling trails.

        Returns
        -------
        nvidia.dali.ops.decoders.ImageRandomCrop
            The operator.

        """
        if isinstance(output_type, str):
            output_type = getattr(types, output_type)
        return ops.decoders.ImageRandomCrop(
            output_type=output_type,
            host_memory_padding=host_memory_padding,
            device_memory_padding=device_memory_padding,
            random_area=random_area,
            random_aspect_ratio=random_aspect_ratio,
            num_attempts=num_attempts,
            device=context.get_device_type(mixed=True),
            **kwargs
        )
