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
"""Reshaping layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import array_ops
from dragon.core.ops import vision_ops
from dragon.core.util import nest
from dragon.vm.keras.core.engine.base_layer import Layer
from dragon.vm.keras.core.engine.input_spec import InputSpec
from dragon.vm.keras.core.utils import conv_utils


class Flatten(Layer):
    """Layer to reshape input into a matrix.

    Examples:

    ```python
    # Reshape an input taking 2 dimensions or more
    m = tf.keras.layers.Flatten()
    x2d = m(tf.ones([24, 1]))  # (24, 1)
    x4d = m(tf.ones([1, 2, 3, 4]))  # (1, 24)

    # Set the ``data_format`` to 'channels_first'
    # will transpose the input before flattening
    mm = tf.keras.layers.Flatten(data_format='channels_first')
    x = tf.random.uniform([1, 2, 3])
    print(m(x))
    print(mm(x))
    ```

    """

    def __init__(self, data_format=None, **kwargs):
        """Create a ``Flatten`` layer.

        Parameters
        ----------
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.

        """
        super(Flatten, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=1)

    def call(self, inputs):
        return array_ops.flatten(inputs, axis=1)


class Permute(Layer):
    """Layer to permute the dimensions of input.

    Examples:

    ```python
    x = tf.random.uniform((2, 1, 3))

    # (2, 1, 3) => (2, 3, 1)
    # Note that the dimensions should start from axis 1
    print(tf.keras.layers.Permute((2, 1))(x))
    ```

    """

    def __init__(self, dims, **kwargs):
        """Create a ``Permute`` layer.

        Parameters
        ----------
        dims : Sequence[int]
            The output dimension order.

        """
        super(Permute, self).__init__(**kwargs)
        self.dims = nest.flatten(dims)
        if sorted(dims) != list(range(1, len(dims) + 1)):
            raise ValueError(
                'Argument <dims> should be consecutive and start from 1.\n'
                'Got {}'.format(str(dims)))
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)

    def call(self, inputs):
        return array_ops.transpose(inputs, perm=[0] + self.dims)


class Reshape(Layer):
    """Layer to change the dimensions of input.

    Examples:

    ```python
    x = tf.random.uniform((2, 1, 3))

    # (2, 1, 3) => (2, 3)
    # Note that the dimensions should start from axis 1
    print(tf.keras.layers.Reshape([3])(x))

    # (2, 1, 3) => (2, 3)
    # At most one dimension could be set to ``-1``
    # to infer remain elements
    print(tf.keras.layers.Reshape([-1])(x))

    # (2, 1, 3) => (2, 1, 3)
    # Set dimension to ``0`` will keep it unchanged
    print(tf.keras.layers.Reshape([0, -1])(x))
    ```
    """

    def __init__(self, target_shape, **kwargs):
        """Create a ``Reshape`` layer.

        Parameters
        ----------
        target_shape : Sequence[int]
            The output dimensions.

        """
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = nest.flatten(target_shape)

    def call(self, inputs):
        return array_ops.reshape(inputs, shape=[0] + self.target_shape)


class UpSampling(Layer):
    """Upsampling layer."""

    def __init__(
        self,
        rank,
        size=2,
        data_format='channels_last',
        interpolation='nearest',
        **kwargs
    ):
        """Create an ``Upsampling`` Layer.

        Parameters
        ----------
        rank : int
            The number of spatial axes.
        size : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        interpolation : str, optional, default='nearest'
            ``'nearest'`` or ``'linear'``.

        """
        super(UpSampling, self).__init__(**kwargs)
        self.rank = rank
        self.size = conv_utils.normalize_tuple(size, rank)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=rank + 2)

    def call(self, inputs):
        return vision_ops.resize(
            inputs,
            scales=[float(x) for x in self.size],
            mode=self.interpolation,
            data_format=conv_utils.convert_data_format(self.data_format),
        )


class UpSampling1D(UpSampling):
    """1D upsampling layer."""

    def __init__(
        self,
        size=2,
        data_format='channels_last',
        interpolation='nearest',
        **kwargs
    ):
        """Create an ``Upsampling1D`` Layer.

        Parameters
        ----------
        size : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        interpolation : str, optional, default='nearest'
            ``'nearest'`` or ``'linear'``.

        """
        super(UpSampling1D, self).__init__(
            rank=1,
            size=size,
            data_format=data_format,
            interpolation=interpolation,
            **kwargs
        )


class UpSampling2D(UpSampling):
    """2D upsampling layer."""

    def __init__(
        self,
        size=2,
        data_format='channels_last',
        interpolation='nearest',
        **kwargs
    ):
        """Create an ``Upsampling2D`` Layer.

        Parameters
        ----------
        size : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        interpolation : str, optional, default='nearest'
            ``'nearest'`` or ``'bilinear'``.

        """
        super(UpSampling2D, self).__init__(
            rank=2,
            size=size,
            data_format=data_format,
            interpolation=interpolation.replace('bilinear', 'linear'),
            **kwargs
        )


class UpSampling3D(UpSampling):
    """3D upsampling layer."""

    def __init__(
        self,
        size=2,
        data_format='channels_last',
        interpolation='nearest',
        **kwargs
    ):
        """Create an ``Upsampling3D`` Layer.

        Parameters
        ----------
        size : Union[number, Sequence[number]], optional
            The scale factor along each input dimension.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        interpolation : str, optional, default='nearest'
            ``'nearest'`` or ``'trilinear'``.

        """
        super(UpSampling3D, self).__init__(
            rank=3,
            size=size,
            data_format=data_format,
            interpolation=interpolation.replace('trilinear', 'linear'),
            **kwargs
        )


class ZeroPadding(Layer):
    """Zero padding layer."""

    def __init__(self, rank, padding=1, data_format='channels_last', **kwargs):
        """Create a ``ZeroPadding`` Layer.

        Parameters
        ----------
        rank : int
            The number of spatial axes.
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding, self).__init__(**kwargs)
        self.rank = rank
        self.padding = conv_utils.normalize_paddings(padding, rank)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if self.data_format == 'channels_first':
            self.padding = conv_utils.normalize_paddings(0, 2) + self.padding
        else:
            self.padding = conv_utils.normalize_paddings(0, 1) + self.padding
            self.padding += conv_utils.normalize_paddings(0, 1)
        self.input_spec = InputSpec(ndim=rank + 2)

    def call(self, inputs):
        return array_ops.pad(inputs, pads=self.padding)


class ZeroPadding1D(ZeroPadding):
    """1D zero padding layer."""

    def __init__(self, padding=1, data_format='channels_last', **kwargs):
        """Create a ``ZeroPadding1D`` Layer.

        Parameters
        ----------
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding1D, self).__init__(
            rank=1, padding=padding, data_format=data_format, **kwargs)


class ZeroPadding2D(ZeroPadding):
    """2D zero padding layer."""

    def __init__(self, padding=1, data_format='channels_last', **kwargs):
        """Create a ``ZeroPadding2D`` Layer.

        Parameters
        ----------
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding2D, self).__init__(
            rank=2, padding=padding, data_format=data_format, **kwargs)


class ZeroPadding3D(ZeroPadding):
    """3D zero padding layer."""

    def __init__(self, padding=1, data_format='channels_last', **kwargs):
        """Create an ``ZeroPadding3D`` Layer.

        Parameters
        ----------
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding3D, self).__init__(
            rank=3, padding=padding, data_format=data_format, **kwargs)
