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
"""Math ops."""

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


class Normalize(object):
    """Normalize input.

    Examples:

    ```python
    norm = dali.ops.Normalize(
        # Batch normalization case of HWC layout
        axes=(0, 1), batch=True, epsilon=1e-5,
    )
    y = norm(inputs['x'])
    ```

    """

    def __new__(
        cls,
        axes=(0, 1),
        mean=None,
        stddev=None,
        scale=1.0,
        shift=0.0,
        batch=False,
        epsilon=0,
        dtype='float32',
        **kwargs
    ):
        """Create a ``Normalize`` operator.

        Parameters
        ----------
        axes : Sequence[int], optional
            The axes to normalize.
        mean : float, optional
            The value to subtract.
        stddev : float, optional
            The value to divide after subtraction.
        scale : float, optional, default=1.0
            The scale factor after normalization.
        shift : float, optional, default=0.0
            The shift factor after normalization.
        batch : bool, optional, default=False
            Whether to compute mean and stddev across the batch.
        epsilon : float, optional, default=0
            The value added to the computed variance.
        dtype : str, optional, default='float32'
            The output data type.

        Returns
        -------
        nvidia.dali.ops.Normalize
            The operator.

        """
        if isinstance(dtype, six.string_types):
            dtype = getattr(types, dtype.upper())
        return ops.Normalize(
            axes=axes,
            mean=mean,
            stddev=stddev,
            scale=scale,
            shift=shift,
            batch=batch,
            epsilon=epsilon,
            dtype=dtype,
            device=context.get_device_type(),
            **kwargs
        )
