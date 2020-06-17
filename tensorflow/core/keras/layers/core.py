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
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import activation_ops
from dragon.core.ops import array_ops
from dragon.core.ops import math_ops
from dragon.core.util import nest
from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.framework import tensor_shape
from dragon.vm.tensorflow.core.keras import activations
from dragon.vm.tensorflow.core.keras import initializers
from dragon.vm.tensorflow.core.keras import regularizers
from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer
from dragon.vm.tensorflow.core.keras.engine.input_spec import InputSpec
from dragon.vm.tensorflow.core.keras.utils import conv_utils


class Dense(Layer):
    """The fully-connected layer."""

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        """Create a ``Dense`` layer.

        Parameters
        ----------
        units : int
            The number of output units.
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
        super(Dense, self).__init__(**kwargs)
        self.input_dim = kwargs.get('input_dim', None)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or dtypes.float32)
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                'Unable to build `Dense` layer with non-floating point '
                'dtype %s' % (dtype,)
            )
        if self.input_dim is None:
            input_shape = tensor_shape.TensorShape(input_shape)
            if tensor_shape.dimension_value(input_shape[-1]) is None:
                raise ValueError(
                    'The last dimension of the inputs should be defined.\n'
                    'Or you should specify <input_dim> in the constructor.'
                )
            last_dim = tensor_shape.dimension_value(input_shape[-1])
        else:
            last_dim = self.input_dim
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        outputs = math_ops.fully_connected(
            [inputs, self.kernel] + [self.bias]
            if self.use_bias else [],
            axis=-1,
            transW=False,
        )
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class Dropout(Layer):
    r"""The dropout layer.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * \text{Bernoulli}(p=1 - prob)

    Examples:

    ```python
    m = tf.keras.layers.Dropout(0.5)
    print(m(tf.ones((2, 3), 'float32')))
    ```

    """

    def __init__(self, rate, **kwargs):
        """Create a ``Dropout`` layer.

        Parameters
        ----------
        rate : Union[float, dragon.Tensor]
            The dropping probability.

        """
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        if self.trainable:
            return activation_ops.dropout(
                inputs,
                prob=self.rate,
                inplace=self.inplace,
            )
        return inputs


class Flatten(Layer):
    """The layer to reshape input into a matrix.

    Examples:

    ```python
    # Reshape an input taking any dimensions
    m = tf.keras.layers.Flatten()
    x1d = m(tf.ones([24]))  # (24, 1)
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
        if self.data_format == 'channels_first':
            perm = [0] + [i for i in range(2, len(inputs.shape))] + [1]
            inputs = array_ops.transpose(inputs, perm=perm)
        return array_ops.flatten(inputs, keep_axes=2)


class Permute(Layer):
    """The layer to permute the dimensions of input.

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
                'Got {}'.format(str(dims))
            )
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)

    def call(self, inputs):
        return array_ops.transpose(inputs, perm=[0] + self.dims)


class Reshape(Layer):
    """The layer to change the dimensions of input.

    Examples:

    ```python
    x = tf.random.uniform((2, 1, 3))

    # (2, 1, 3) => (2, 3)
    # Note that the dimensions should start from axis 1
    print(tf.keras.layers.Reshape([3])(x))

    # (2, 1, 3) => (2, 3)
    # At most one dimension could be set to ``-1``
    # to infer the remain elements
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
