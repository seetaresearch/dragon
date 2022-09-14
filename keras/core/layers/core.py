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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import activation_ops
from dragon.core.ops import math_ops
from dragon.vm.keras.core import activations
from dragon.vm.keras.core import initializers
from dragon.vm.keras.core import regularizers
from dragon.vm.keras.core.engine.base_layer import Layer
from dragon.vm.keras.core.engine.input_spec import InputSpec
from dragon.vm.tensorflow.core.framework import dtypes


class Activation(Layer):
    """Activation layer."""

    def __init__(self, activation, **kwargs):
        """Create an ``Activation`` layer.

        Parameters
        ----------
        activation : Union[callable, str], optional
            The optional activation function.

        """
        super(Activation, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        if self.inplace:
            return self.activation(inputs, inplace=True)
        return self.activation(inputs)


class Dense(Layer):
    """Fully-connected layer."""

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
            ``True`` to apply a ``bias``.
        kernel_initializer : Union[callable, str], optional
            The initializer for kernel tensor.
        bias_initializer : Union[callable, str], optional
            The initializer for bias tensor.
        kernel_regularizer : Union[callable, str], optional
            The regularizer for kernel tensor.
        bias_regularizer : Union[callable, str], optional
            The regularizer for bias tensor.

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
                'dtype %s' % (dtype,))
        last_dim = int(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        outputs = math_ops.gemm(
            [inputs, self.kernel] +
            ([self.bias] if self.use_bias else []))
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class Dropout(Layer):
    r"""Layer to apply the dropout function.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * (r \sim \mathcal{B}(1, 1 - \text{rate}))

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
            The dropping ratio.

        """
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs, training=None):
        if training:
            return activation_ops.dropout(
                inputs, ratio=self.rate, inplace=self.inplace)
        return inputs
