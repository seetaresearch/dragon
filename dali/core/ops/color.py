# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
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


class BrightnessContrast(object):
    """Adjust the brightness and contrast.

    Examples:

    ```python
    # Historical jitter range for brightness and contrast
    twist_rng = dali.ops.Uniform(range=[0.6, 1.4])

    bc = dali.ops.BrightnessContrast()
    y = bc(inputs['x'], brightness=twist_rng(), contrast=twist_rng())
    ```

    """

    def __new__(cls):
        """Create a ``BrightnessContrastBrightnessContrast`` operator.

        Returns
        -------
        nvidia.dali.ops.BrightnessContrast
            The operator.

        """
        return ops.BrightnessContrast(device=context.get_device_type())


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

    def __new__(cls):
        """Create a ``Hsv`` operator.

        Returns
        -------
        nvidia.dali.ops.Hsv
            The operator.

        """
        return ops.Hsv(device=context.get_device_type())
