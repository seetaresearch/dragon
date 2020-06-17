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


class Cast(object):
    """Cast the data type of input.

    Examples:

    ```python
    cast = dali.ops.Cast(dtype='int64')
    y = cast(inputs['x'])
    ```

    """

    def __new__(cls, dtype):
        """Create a ``Cast`` operator.

        Parameters
        ----------
        dtype : str, optional
            The output data type.

        Returns
        -------
        nvidia.dali.ops.Cast
            The operator.

        """
        if isinstance(dtype, six.string_types):
            dtype = getattr(types, dtype.upper())
        return ops.Cast(
            dtype=dtype,
            device=context.get_device_type(),
        )


class Pad(object):
    """Pad input to have the same dimensions.

    Examples:

    ```python
    pad = dali.ops.Pad(
        # The axes to pad
        axes=[0, 1],
        # The constant value fill at the right side
        fill_value=0.,
    )
    y = pad(inputs['x'])
    ```

    """

    def __new__(cls, axes=(0, 1), fill_value=0., align=None):
        """Create a ``Pad`` operator.

        Parameters
        ----------
        axes : Sequence[int], optional
            The padding axes.
        fill_value : number, optional, default=0.
            The constant padding value.
        align : Union[int, Sequence[int]], optional
            The size to align the padding shape.

        Returns
        -------
        nvidia.dali.ops.Pad
            The operator.

        """
        return ops.Pad(
            axes=axes,
            fill_value=fill_value,
            align=align,
            device=context.get_device_type(),
        )


class Reshape(object):
    """Change the dimensions of input.

    Examples:

    ```python
    # Reshape to a constant shape
    reshape1 = dali.ops.Reshape(shape=(2, 3))
    y = reshape1(inputs['x'])

    # Reshape to the shape given by another tensor
    reshape2 = dali.ops.Reshape()
    z = reshape2(inputs['x'], inputs['x_shape'])
    ```

    """

    def __new__(cls, shape=None):
        """Create a ``Reshape`` operator.

        Parameters
        ----------
        shape : Sequence[int], optional
            The optional output shape.

        Returns
        -------
        nvidia.dali.ops.Reshape
            The operator.

        """
        return ops.Reshape(
            shape=shape,
            device=context.get_device_type(),
        )
