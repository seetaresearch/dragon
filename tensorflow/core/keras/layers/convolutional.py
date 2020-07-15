# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorflow.core.framework import tensor_shape
from dragon.vm.tensorflow.core.keras import activations
from dragon.vm.tensorflow.core.keras import initializers
from dragon.vm.tensorflow.core.keras import regularizers
from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer
from dragon.vm.tensorflow.core.keras.engine.input_spec import InputSpec
from dragon.vm.tensorflow.core.keras.utils import conv_utils
from dragon.vm.tensorflow.core.ops import array_ops
from dragon.vm.tensorflow.core.ops import nn_ops


class Conv(Layer):
    """The generic convolution layer."""

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super(Conv, self).__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank)
        self.strides = conv_utils.normalize_tuple(strides, rank)
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.conv_function = kwargs.get('conv_function', nn_ops.convolution)
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the input '
                'should be determined, got None.')
        input_dim = int(input_shape[channel_axis])
        # Assume that kernel is packed into NCHW format
        # for computing the fans correctly
        if self.filters > 0:
            kernel_shape = (self.filters, input_dim) + self.kernel_size
        else:
            self.filters = input_dim
            kernel_shape = (input_dim, 1) + self.kernel_size
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        outputs = self.conv_function(
            input=inputs,
            filters=self.kernel,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            data_format=conv_utils.convert_data_format(
                self.data_format,
                self.rank + 2
            ),
        )
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn_ops.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn_ops.bias_add(outputs, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _get_channel_axis(self):
        return 1 if self.data_format == 'channels_first' else -1


class Conv2D(Conv):
    """The 2d convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Conv2D`` Layer.

        Parameters
        ----------
        filters : int
            The number of output filters.
        kernel_size : Sequence[int]
            The shape of convolution kernel.
        strides : Sequence[int], optional, default=1
            The stride(s) of sliding window.
        padding : Union[{'valid', 'same'}, Sequence[int]], optional
            The padding algorithm or padding sizes.
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.
        dilation_rate : Sequence[int], optional
            The rate(s) of dilated kernel.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            **True** to apply a ``bias``.
        kernel_initializer : Union[callable, str], optional
            The initializer for ``kernel``.
        bias_initializer : Union[callable, str], optional
            The initializer for ``bias``.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for ``kernel``.
        bias_regularizer : Union[callable, str], optional
            The regularizer for ``bias``.

        """
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            **kwargs
        )


class Conv2DTranspose(Conv2D):
    """The 2d deconvolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Conv2DTranspose`` Layer.

        Parameters
        ----------
        filters : int
            The number of output filters.
        kernel_size : Sequence[int]
            The shape of convolution kernel.
        strides : Sequence[int], optional
            The stride(s) of sliding window.
        padding : Union[{'valid', 'same'}, Sequence[int]], optional
            The padding algorithm or padding sizes.
        output_padding : Sequence[int], optional
            The sizes of padded to the output.
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.
        dilation_rate : Sequence[int], optional
            The rate(s) of dilated kernel.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            **True** to apply a ``bias``.
        kernel_initializer : Union[callable, str], optional
            The initializer for ``kernel``.
        bias_initializer : Union[callable, str], optional
            The initializer for ``bias``.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for ``kernel``.
        bias_regularizer : Union[callable, str], optional
            The regularizer for ``bias``.

        """
        super(Conv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            **kwargs
        )
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, self.rank)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the inputs '
                'should be determined, got None.'
            )
        input_dim = int(input_shape[channel_axis])
        # Assume that kernel is packed into NCHW format,
        # for computing the fans correctly.
        kernel_shape = (input_dim, self.filters) + self.kernel_size
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2
        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding
        out_height = conv_utils.deconv_output_length(
            height,
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
        outputs = nn_ops.conv_transpose(
            input=inputs,
            filters=self.kernel,
            output_shape=output_shape,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            data_format=conv_utils.convert_data_format(
                self.data_format,
                self.rank + 2
            ),
        )
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn_ops.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn_ops.bias_add(outputs, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class DepthwiseConv2D(Conv2D):
    r"""The 2d depthwise convolution layer.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``DepthwiseConv2D`` Layer.

        Parameters
        ----------
        filters : int
            The number of output filters.
        kernel_size : Sequence[int]
            The shape of convolution kernel.
        strides : Sequence[int], optional, default=1
            The stride(s) of sliding window.
        padding : Union[{'valid', 'same'}, Sequence[int]], optional
            The padding algorithm or padding sizes.
        data_format : {'channels_first', 'channels_last'}, optional
            The optional data format.
        dilation_rate : Sequence[int], optional
            The rate(s) of dilated kernel.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            **True** to apply a ``bias``.
        kernel_initializer : Union[callable, str], optional
            The initializer for ``kernel``.
        bias_initializer : Union[callable, str], optional
            The initializer for ``bias``.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for ``kernel``.
        bias_regularizer : Union[callable, str], optional
            The regularizer for ``bias``.

        """
        super(DepthwiseConv2D, self).__init__(
            filters=0,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            conv_function=nn_ops.depthwise_conv2d,
            **kwargs
        )
