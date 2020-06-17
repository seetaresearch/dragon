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

from dragon.core.util import six
from dragon.vm.dali.core import context
from dragon.vm.dali.core import types


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
        interp_type='TRIANGULAR',
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
        interp_type : {'NN', 'LINEAR', 'TRIANGULAR', 'CUBIC', 'GAUSSIAN', 'LANCZOS3'}, optional
            The interpolation method.

        """
        if isinstance(interp_type, six.string_types):
            interp_type = getattr(types, 'INTERP_' + interp_type.upper())
        return ops.Resize(
            resize_x=resize_x,
            resize_y=resize_y,
            resize_shorter=resize_shorter,
            resize_longer=resize_longer,
            max_size=max_size,
            interp_type=interp_type,
            device=context.get_device_type(),
        )
