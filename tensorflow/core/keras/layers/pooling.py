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
from dragon.vm.tensorflow.core.ops import nn
from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer
from dragon.vm.tensorflow.core.keras.engine.input_spec import InputSpec
from dragon.vm.tensorflow.core.keras.utils import conv_utils


class Pooling2D(Layer):
    """The generic 2d pooling layer."""

    def __init__(
        self,
        pool_function,
        pool_size,
        strides,
        padding='valid',
        data_format=None,
        name=None,
        **kwargs
    ):
        super(Pooling2D, self).__init__(name=name, **kwargs)
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2)
        self.strides = conv_utils.normalize_tuple(strides, 2)
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding,
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        return outputs


class GlobalPooling2D(Layer):
    """The generic 2d global pooling layer."""

    def __init__(self, pool_function, data_format=None, **kwargs):
        super(GlobalPooling2D, self).__init__(**kwargs)
        self.pool_function = pool_function
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        outputs = self.pool_function(
            inputs,
            kernel_shape=1,
            strides=1,
            pads=0,
            global_pooling=True,
            data_format=conv_utils.convert_data_format(self.data_format, 4),
        )
        return outputs


class AveragePooling2D(Pooling2D):
    """The average 2d pooling layer."""

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding='valid',
        data_format=None,
        **kwargs
    ):
        """Create a ``MaxPooling2D`` Layer.

        Parameters
        ----------
        pool_size : Sequence[int]
            The size(s) of sliding window.
        strides : Sequence[int], optional
            The stride(s) of sliding window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or padding sizes.
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.

        """
        super(AveragePooling2D, self).__init__(
            nn.avg_pool2d,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class GlobalAveragePooling2D(GlobalPooling2D):
    """The global average 2d pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalAveragePooling2D`` Layer.

        Parameters
        ----------
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.

        """
        super(GlobalAveragePooling2D, self).__init__(
            functools.partial(vision_ops.pool2d, mode='AVG'),
            data_format=data_format,
            **kwargs
        )


class GlobalMaxPooling2D(GlobalPooling2D):
    """The global max 2d pooling layer."""

    def __init__(self, data_format=None, **kwargs):
        """Create a ``GlobalMaxPooling2D`` Layer.

        Parameters
        ----------
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.

        """
        super(GlobalMaxPooling2D, self).__init__(
            functools.partial(vision_ops.pool2d, mode='MAX'),
            data_format=data_format,
            **kwargs
        )


class MaxPooling2D(Pooling2D):
    """The max 2d pooling layer."""

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding='valid',
        data_format=None,
        **kwargs
    ):
        """Create a ``MaxPooling2D`` Layer.

        Parameters
        ----------
        pool_size : Sequence[int]
            The size(s) of sliding window.
        strides : Sequence[int], optional
            The stride(s) of sliding window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or padding sizes.
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.

        """
        super(MaxPooling2D, self).__init__(
            nn.max_pool2d,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


# Aliases
AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalMaxPool2D = GlobalMaxPooling2D
