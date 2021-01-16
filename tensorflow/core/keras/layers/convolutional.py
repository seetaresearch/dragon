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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import vision_ops
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
    """Convolution layer."""

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format='channels_last',
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
        """Create a ``Conv`` Layer.

        Parameters
        ----------
        rank : int
            The number of spatial axes.
        filters : int, optional
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[int, Sequence[int], str], optional
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
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
            raise ValueError('The channel dimension of the input '
                             'should be determined, got None.')
        input_dim = int(input_shape[channel_axis])
        if self.filters is None:
            input_dim, self.filters = 1, input_dim
        kernel_shape = [self.filters] + list(self.kernel_size)
        if self.data_format == 'channels_first':
            kernel_shape.insert(1, input_dim)
        else:
            kernel_shape.append(input_dim)
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
        output = self.conv_function(
            input=inputs,
            filters=self.kernel,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            data_format=conv_utils.convert_data_format(
                self.data_format, self.rank + 2))
        if self.use_bias:
            output = self._add_bias(output)
        if self.activation is not None:
            return self.activation(output)
        return output

    def _add_bias(self, input):
        """Add a bias tensor to input."""
        if self.data_format == 'channels_first':
            if self.rank == 1:
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                input += bias
                output = input
            else:
                output = nn_ops.bias_add(input, self.bias, data_format='NCHW')
        else:
            output = nn_ops.bias_add(input, self.bias, data_format='NHWC')
        return output

    def _get_channel_axis(self):
        """Return the channel axis."""
        return 1 if self.data_format == 'channels_first' else -1


class Conv1D(Conv):
    """1D convolution layer."""

    def __init__(
        self,
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
        **kwargs
    ):
        """Create a ``Conv1D`` Layer.

        Parameters
        ----------
        filters : int`
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(Conv1D, self).__init__(
            rank=1,
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
            **kwargs)


class Conv2D(Conv):
    """2D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
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
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

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
            **kwargs)


class Conv3D(Conv):
    """3D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Conv3D`` Layer.

        Parameters
        ----------
        filters : int
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(Conv3D, self).__init__(
            rank=3,
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
            **kwargs)


class ConvTranspose(Conv):
    """Deconvolution layer."""

    def __init__(
        self,
        rank,
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
        """Create a ``ConvTranspose`` Layer.

        Parameters
        ----------
        filters : int
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        output_padding : Sequence[int], optional
            The additional size added to the output shape.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(ConvTranspose, self).__init__(
            rank=rank,
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
            **kwargs)
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, self.rank)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be determined, got None.')
        input_dim = int(input_shape[channel_axis])
        if self.filters is None:
            input_dim, self.filters = 1, input_dim
        kernel_shape = [input_dim] + list(self.kernel_size)
        if self.data_format == 'channels_first':
            kernel_shape.insert(1, self.filters)
        else:
            kernel_shape.append(self.filters)
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
        output = nn_ops.conv_transpose(
            input=inputs,
            filters=self.kernel,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            output_shape=None,
            dilations=self.dilation_rate,
            data_format=conv_utils.convert_data_format(
                self.data_format, self.rank + 2),
        )
        if self.use_bias:
            self._add_bias(output)
        if self.activation is not None:
            return self.activation(output)
        return output


class Conv1DTranspose(ConvTranspose):
    """1D deconvolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Conv1DTranspose`` Layer.

        Parameters
        ----------
        filters : int`
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        output_padding : Sequence[int], optional
            The additional size added to the output shape.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(Conv1DTranspose, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            **kwargs)


class Conv2DTranspose(ConvTranspose):
    """2D deconvolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Conv1DTranspose`` Layer.

        Parameters
        ----------
        filters : int`
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        output_padding : Sequence[int], optional
            The additional size added to the output shape.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(Conv2DTranspose, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            **kwargs)


class Conv3DTranspose(ConvTranspose):
    """3D deconvolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Conv1DTranspose`` Layer.

        Parameters
        ----------
        filters : int`
            The number of output filters.
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        output_padding : Sequence[int], optional
            The additional size added to the output shape.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(Conv3DTranspose, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            **kwargs)


class DepthwiseConv2D(Conv):
    r"""2D depthwise convolution layer.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.
    """

    def __init__(
        self,
        kernel_size,
        strides=1,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
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
        kernel_size : Union[int, Sequence[int]]
            The shape of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        padding : Union[str, Sequence[int]], optional
            The padding algorithm or size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Union[int, Sequence[int]], optional, default=1
            The rate of dilated convolution.
        activation : Union[callable, str], optional
            The optional activation function.
        use_bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

        """
        super(DepthwiseConv2D, self).__init__(
            rank=2,
            filters=None,
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

    def __init__(
        self,
        rank,
        padding=1,
        data_format='channels_last',
        **kwargs
    ):
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
        if self.data_format == 'channel_first':
            self.padding = conv_utils.normalize_paddings(0, 2) + self.padding
        else:
            self.padding = conv_utils.normalize_paddings(0, 1) + self.padding
            self.padding = self.padding + conv_utils.normalize_paddings(0, 1)
        self.input_spec = InputSpec(ndim=rank + 2)

    def call(self, inputs):
        return array_ops.pad(inputs, paddings=self.padding)


class ZeroPadding1D(ZeroPadding):
    """1D zero padding layer."""

    def __init__(
        self,
        padding=1,
        data_format='channels_last',
        **kwargs
    ):
        """Create a ``ZeroPadding1D`` Layer.

        Parameters
        ----------
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding1D, self).__init__(
            rank=1,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class ZeroPadding2D(ZeroPadding):
    """2D zero padding layer."""

    def __init__(
        self,
        padding=1,
        data_format='channels_last',
        **kwargs
    ):
        """Create a ``ZeroPadding2D`` Layer.

        Parameters
        ----------
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding2D, self).__init__(
            rank=2,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class ZeroPadding3D(ZeroPadding):
    """3D zero padding layer."""

    def __init__(
        self,
        padding=1,
        data_format='channels_last',
        **kwargs
    ):
        """Create an ``ZeroPadding3D`` Layer.

        Parameters
        ----------
        padding : Union[int, Sequence[int], Sequence[Tuple[int]], optional, default=1
            The padding size.
        data_format : str, optional, default='channels_last'
            ``'channels_first'`` or ``'channels_last'``.

        """
        super(ZeroPadding3D, self).__init__(
            rank=3,
            padding=padding,
            data_format=data_format,
            **kwargs
        )
