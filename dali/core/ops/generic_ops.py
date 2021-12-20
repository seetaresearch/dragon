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
"""Generic ops."""

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


class Cast(object):
    """Cast the data type of input.

    Examples:

    ```python
    cast = dali.ops.Cast(dtype='int64')
    y = cast(inputs['x'])
    ```

    """

    def __new__(cls, dtype, **kwargs):
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
            **kwargs
        )


class Erase(object):
    """Erase regions from the input.

    Examples:

    ```python
    erase = dali.ops.Erase(
        # The axes to erase
        axes=[0, 1],
        # The value fill
        fill_value=0.,
    )
    y = erase(inputs['x'], anchor=(0, 0), shape=(100, 100))
    ```

    """

    def __new__(
        cls,
        axes=(0, 1),
        fill_value=0,
        normalized_anchor=True,
        normalized_shape=True,
        **kwargs
    ):
        """Create an ``Erase`` operator.

        Parameters
        ----------
        axes : Sequence[int], optional
            The padding axes.
        fill_value : Union[number, Sequence[float]], optional
            The value to fill the erased regions.
        normalized_anchor : bool, optional, default=True
            Provided anchor is normalized or not.
        normalized_shape : bool, optional, default=True
            Provided shape is normalized or not.

        Returns
        -------
        nvidia.dali.ops.Erase
            The operator.

        """
        return ops.Erase(
            axes=axes,
            fill_value=fill_value,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
            device=context.get_device_type(),
            **kwargs
        )


class Flip(object):
    """Flip input in selected dimensions.

    Examples:

    ```python
    flip_rng = dali.ops.CoinFlip(0.5)
    flip = dali.ops.Flip()
    y = flip(inputs['x'], horizontal=flip_rng())
    ```

    """

    def __new__(cls, horizontal=None, vertical=None, depthwise=None, **kwargs):
        """Create a ``Flip`` operator.

        Parameters
        ----------
        horizontal : int, optional
            Whether to apply the horizontal flip.
        vertical : int, optional
            Whether to apply the vertical flip.
        depthwise : bool, optional, default=True
            Whether to apply the depthwise flip.

        Returns
        -------
        nvidia.dali.ops.Flip
            The operator.

        """
        return ops.Flip(
            horizontal=horizontal,
            vertical=vertical,
            depthwise=depthwise,
            device=context.get_device_type(),
            **kwargs
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

    def __new__(cls, axes=(0, 1), fill_value=0, align=None, **kwargs):
        """Create a ``Pad`` operator.

        Parameters
        ----------
        axes : Sequence[int], optional
            The padding axes.
        fill_value : number, optional, default=0
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
            **kwargs
        )


class Reshape(object):
    """Change the dimensions of input.

    Examples:

    ```python
    # Reshape to a constant shape
    reshape1 = dali.ops.Reshape(shape=(2, 3))
    y = reshape1(inputs['x'])

    # Reshape to the shape of other tensor
    reshape2 = dali.ops.Reshape()
    z = reshape2(inputs['x'], inputs['x_shape'])
    ```

    """

    def __new__(cls, shape=None, **kwargs):
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
            **kwargs
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
        **kwargs
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
            normalized_shape=normalized_shape,
            device=context.get_device_type(),
            **kwargs
        )
