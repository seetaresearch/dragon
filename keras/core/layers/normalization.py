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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/normalization.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import normalization_ops
from dragon.vm.keras.core import initializers
from dragon.vm.keras.core import regularizers
from dragon.vm.keras.core.engine.base_layer import Layer
from dragon.vm.keras.core.engine.input_spec import InputSpec
from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.framework import tensor_shape


class BatchNormalization(Layer):
    r"""Batch normalization layer.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    """

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        name=None,
        **kwargs
    ):
        """Create a ``BatchNormalization`` layer.

        Parameters
        ----------
        axis : int, optional, default=-1
            The channel axis.
        momentum : float, optional, default=0.99
            The decay factor of running average.
        epsilon : float, optional, default=1e-3
            The epsilon value.
        center : bool, optional, default=True
            ``False`` to freeze the ``beta`` anyway.
        scale : bool, optional, default=True
            ``False`` to freeze the ``gamma`` anyway.
        beta_initializer : Union[callable, str], optional
            The initializer for beta tensor.
        gamma_initializer : Union[callable, str], optional
            The initializer for gamma tensor.
        moving_mean_initializer : Union[callable, str], optional
            The initializer for moving mean tensor.
        moving_variance_initializer : Union[callable, str], optional
            The initializer for moving variance tensor.
        beta_regularizer : Union[callable, str], optional
            The regularizer for beta tensor.
        gamma_regularizer : Union[callable, str], optional
            The regularizer for gamma tensor.

        """
        super(BatchNormalization, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta = None
        self.gamma = None
        self.moving_mean = None
        self.moving_variance = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank: ' + str(input_shape))
        ndims = len(input_shape)
        self.axis = ndims + self.axis if self.axis < 0 else self.axis
        if self.axis < 0 or self.axis >= ndims:
            raise ValueError('Invalid axis: %s' % (self.axis,))
        param_shape = [input_shape.dims[self.axis]]
        self.input_spec = InputSpec(ndim=input_shape.ndims,
                                    axes={self.axis: param_shape[0]})
        self.gamma = self.add_weight(
            name='gamma',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=self.scale)
        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=self.center)
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        return normalization_ops.batch_norm(
            [inputs, self.gamma, self.beta,
             self.moving_mean, self.moving_variance],
            axis=self.axis, momentum=self.momentum, epsilon=self.epsilon,
            use_stats=not training)

    @property
    def _param_dtype(self):
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            return dtypes.float32
        else:
            return self.dtype or dtypes.float32


class LayerNormalization(Layer):
    r"""LayerNormalization layer.
    `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

    """

    def __init__(
        self,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        name=None,
        **kwargs
    ):
        """Create a ``LayerNormalization`` layer.

        Parameters
        ----------
        axis : int, optional, default=-1
            The channel axis.
        momentum : float, optional, default=0.99
            The decay factor of running average.
        epsilon : float, optional, default=1e-3
            The epsilon value.
        center : bool, optional, default=True
            ``False`` to freeze the ``beta`` anyway.
        scale : bool, optional, default=True
            ``False`` to freeze the ``gamma`` anyway.
        beta_initializer : Union[callable, str], optional
            The initializer for beta tensor.
        gamma_initializer : Union[callable, str], optional
            The initializer for gamma tensor.
        beta_regularizer : Union[callable, str], optional
            The regularizer for beta tensor.
        gamma_regularizer : Union[callable, str], optional
            The regularizer for gamma tensor.

        """
        super(LayerNormalization, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank: ' + str(input_shape))
        ndims = len(input_shape)
        self.axis = ndims + self.axis if self.axis < 0 else self.axis
        if self.axis < 0 or self.axis >= ndims:
            raise ValueError('Invalid axis: %s' % (self.axis,))
        param_shape = [input_shape.dims[self.axis]]
        self.input_spec = InputSpec(ndim=input_shape.ndims,
                                    axes={self.axis: param_shape[0]})
        self.gamma = self.add_weight(
            name='gamma',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=self.scale)
        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=self.center)
        self.built = True

    def call(self, inputs):
        return normalization_ops.layer_norm(
            [inputs, self.gamma, self.beta], axis=self.axis,
            epsilon=self.epsilon)

    @property
    def _param_dtype(self):
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            return dtypes.float32
        else:
            return self.dtype or dtypes.float32
