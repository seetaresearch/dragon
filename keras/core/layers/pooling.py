# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/pooling.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.core.ops import vision_ops
from dragon.vm.keras.core.engine.base_layer import Layer
from dragon.vm.keras.core.engine.input_spec import InputSpec
from dragon.vm.keras.core.utils import conv_utils


class Pooling(Layer):
    """Pooling layer."""

    def __init__(
        self,
        rank,
        pool_function,
        pool_size,
        strides,
        padding='valid',
        data_format=None,
        name=None,
        **kwargs
    ):
        super(Pooling, self).__init__(name=name, **kwargs)
        if strides is None:
            strides = pool_size
        self.rank = rank
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, rank)
        self.strides = conv_utils.normalize_tuple(strides, rank)
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=rank + 2)

    def call(self, inputs):
        pads, padding = 0, self.padding
        if not isinstance(self.padding, str):
            pads, padding = self.padding, 'valid'
        return self.pool_function(
            inputs,
            kernel_shape=self.pool_size,
            strides=self.strides,
            pads=pads,
            padding=padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format),
        )


class AveragePooling1D(Pooling):
    """1D average pooling layer."""

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding='valid',
        data_format='channels_last',
        **kwargs
    ):
        """Create a ``AveragePooling1D`` Layer.

        Parameters
        ----------
        pool_size : Union[int, Sequence[int]], optional, default=2
            The size of pooling window.
        strides : Sequence[int], optional
            The stride of pooling window.
        padding : Union[int, Sequence[int], str], optional, default='valid'
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(AveragePooling1D, self).__init__(
            rank=1,
            pool_function=functools.partial(vision_ops.pool, mode='avg'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class AveragePooling2D(Pooling):
    """2D average pooling layer."""

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding='valid',
        data_format='channels_last',
        **kwargs
    ):
        """Create a ``AveragePooling2D`` Layer.

        Parameters
        ----------
        pool_size : Union[int, Sequence[int]], optional, default=2
            The size of pooling window.
        strides : Sequence[int], optional
            The stride of pooling window.
        padding : Union[int, Sequence[int], str], optional, default='valid'
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(AveragePooling2D, self).__init__(
            rank=2,
            pool_function=functools.partial(vision_ops.pool, mode='avg'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class AveragePooling3D(Pooling):
    """3D average pooling layer."""

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding='valid',
        data_format='channels_last',
        **kwargs
    ):
        """Create a ``AveragePooling3D`` Layer.

        Parameters
        ----------
        pool_size : Union[int, Sequence[int]], optional, default=2
            The size of pooling window.
        strides : Sequence[int], optional
            The stride of pooling window.
        padding : Union[int, Sequence[int], str], optional, default='valid'
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(AveragePooling3D, self).__init__(
            rank=3,
            pool_function=functools.partial(vision_ops.pool, mode='avg'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class GlobalAveragePooling1D(Pooling):
    """1D global average pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalAveragePooling1D`` Layer.

        Parameters
        ----------
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(GlobalAveragePooling1D, self).__init__(
            rank=1,
            pool_function=functools.partial(
                vision_ops.pool1d, mode='avg', global_pool=True),
            pool_size=0,
            strides=1,
            padding='valid',
            data_format=data_format,
            **kwargs
        )


class GlobalAveragePooling2D(Pooling):
    """2D global average pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalAveragePooling2D`` Layer.

        Parameters
        ----------
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(GlobalAveragePooling2D, self).__init__(
            rank=2,
            pool_function=functools.partial(
                vision_ops.pool2d, mode='avg', global_pool=True),
            pool_size=0,
            strides=1,
            padding='valid',
            data_format=data_format,
            **kwargs
        )


class GlobalAveragePooling3D(Pooling):
    """3D global average pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalAveragePooling3D`` Layer.

        Parameters
        ----------
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(GlobalAveragePooling3D, self).__init__(
            rank=3,
            pool_function=functools.partial(
                vision_ops.pool3d, mode='avg', global_pool=True),
            pool_size=0,
            strides=1,
            padding='valid',
            data_format=data_format,
            **kwargs
        )


class GlobalMaxPooling1D(Pooling):
    """1D global max pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalMaxPooling1D`` Layer.

        Parameters
        ----------
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(GlobalMaxPooling1D, self).__init__(
            rank=1,
            pool_function=functools.partial(
                vision_ops.pool1d, mode='max', global_pool=True),
            pool_size=0,
            strides=1,
            padding='valid',
            data_format=data_format,
            **kwargs
        )


class GlobalMaxPooling2D(Pooling):
    """2D global max pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalMaxPooling2D`` Layer.

        Parameters
        ----------
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(GlobalMaxPooling2D, self).__init__(
            rank=2,
            pool_function=functools.partial(
                vision_ops.pool2d, mode='max', global_pool=True),
            pool_size=0,
            strides=1,
            padding='valid',
            data_format=data_format,
            **kwargs
        )


class GlobalMaxPooling3D(Pooling):
    """3D global max pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalMaxPooling3D`` Layer.

        Parameters
        ----------
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(GlobalMaxPooling3D, self).__init__(
            rank=3,
            pool_function=functools.partial(
                vision_ops.pool3d, mode='max', global_pool=True),
            pool_size=0,
            strides=1,
            padding='valid',
            data_format=data_format,
            **kwargs
        )


class MaxPooling1D(Pooling):
    """1D max pooling layer."""

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding='valid',
        data_format=None,
        **kwargs
    ):
        """Create a ``MaxPooling1D`` Layer.

        Parameters
        ----------
        pool_size : Union[int, Sequence[int]], optional, default=2
            The size of pooling window.
        strides : Sequence[int], optional
            The stride of pooling window.
        padding : Union[int, Sequence[int], str], optional, default='valid'
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(MaxPooling1D, self).__init__(
            rank=1,
            pool_function=functools.partial(vision_ops.pool, mode='max'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class MaxPooling2D(Pooling):
    """2D max pooling layer."""

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding='valid',
        data_format=None,
        **kwargs
    ):
        """Create a ``MaxPooling2D`` Layer.

        Parameters
        ----------
        pool_size : Union[int, Sequence[int]], optional, default=2
            The size of pooling window.
        strides : Sequence[int], optional
            The stride of pooling window.
        padding : Union[int, Sequence[int], str], optional, default='valid'
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(MaxPooling2D, self).__init__(
            rank=2,
            pool_function=functools.partial(vision_ops.pool, mode='max'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class MaxPooling3D(Pooling):
    """3D max pooling layer."""

    def __init__(
        self,
        pool_size=2,
        strides=None,
        padding='valid',
        data_format=None,
        **kwargs
    ):
        """Create a ``MaxPooling3D`` Layer.

        Parameters
        ----------
        pool_size : Union[int, Sequence[int]], optional, default=2
            The size of pooling window.
        strides : Sequence[int], optional
            The stride of pooling window.
        padding : Union[int, Sequence[int], str], optional, default='valid'
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(MaxPooling3D, self).__init__(
            rank=3,
            pool_function=functools.partial(vision_ops.pool, mode='max'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


# Aliases
AvgPool1D = AveragePooling1D
AvgPool2D = AveragePooling2D
AvgPool3D = AveragePooling3D
MaxPool1D = MaxPooling1D
MaxPool2D = MaxPooling2D
MaxPool3D = MaxPooling3D
GlobalAvgPool1D = GlobalAveragePooling1D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalAvgPool3D = GlobalAveragePooling3D
GlobalMaxPool1D = GlobalMaxPooling1D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool3D = GlobalMaxPooling3D
