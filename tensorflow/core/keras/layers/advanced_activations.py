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
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/advanced_activations.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorflow.core.keras import activations
from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer


class ELU(Layer):
    r"""The layer to apply the exponential linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \text{ELU}(x) =
        \begin{cases}
            x, & \text{ if } x \geq 0 \\
            \alpha * (e^{x} - 1), & \text{ otherwise }
        \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.layers.ELU(inplace=False)(x))
    ```

    """

    def __init__(self, alpha=1., **kwargs):
        r"""Create an ``ELU`` layer.

        Parameters
        ----------
        alpha : float, optional, default=0.3
            The value to :math:`\alpha`.

        """
        super(ELU, self).__init__(**kwargs)
        self.alpha = alpha
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        return activations.elu(
            inputs,
            alpha=self.alpha,
            inplace=self.inplace,
        )


class LeakyReLU(Layer):
    r"""The layer to apply the leaky rectified linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
            x, & \text{ if } x \geq 0 \\
            \alpha * x, & \text{ otherwise }
        \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.layers.LeakyReLU(inplace=False)(x))
    ```

    """

    def __init__(self, alpha=0.3, **kwargs):
        r"""Create a ``LeakyReLU`` layer.

        Parameters
        ----------
        alpha : float, optional, default=0.3
            The value to :math:`\alpha`.

        """
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        return activations.relu(
            inputs,
            alpha=self.alpha,
            inplace=self.inplace,
        )


class ReLU(Layer):
    r"""The layer to apply the rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \text{ReLU}(x) =
            \begin{cases}
                \min(x, v_{max}), & \text{ if } x \geq 0 \\
                \alpha * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.layers.ReLU(inplace=False)(x))
    ```

    """

    def __init__(self, max_value=None, negative_slope=0, **kwargs):
        r"""Create a ``ReLU`` layer.

        Parameters
        ----------
        max_value : number, optional
            The value to :math:`v_{max}`.
        negative_slope : float, optional, default=0.
            The value to :math:`\alpha`.

        """
        super(ReLU, self).__init__(**kwargs)
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        return activations.relu(
            inputs,
            alpha=self.negative_slope,
            max_value=self.max_value,
            inplace=self.inplace,
        )


class SELU(Layer):
    r"""The layer to apply the scaled exponential linear unit.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    .. math::
        \text{SELU}(x) = 1.0507 *
        \begin{cases}
            x, & \text{ if } x \geq 0 \\
            1.67326 * (e^{x} - 1), & \text{ otherwise }
        \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.layers.SELU(inplace=False)(x))
    ```

    """

    def __init__(self, **kwargs):
        r"""Create an ``SELU`` layer."""
        super(SELU, self).__init__(**kwargs)
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        return activations.selu(inputs, inplace=self.inplace)


class Softmax(Layer):
    r"""The layer to apply the softmax function.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x) = \frac{e^{x_{i}}}{\sum e^{x_{j}}}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.layers.Softmax(inplace=False)(x))
    ```

    """

    def __init__(self, axis=-1, **kwargs):
        r"""Create a ``Softmax`` layer.

        Parameters
        ----------
        axis : int, optional, default=-1
            The axis to reduce.

        """
        super(Softmax, self).__init__(**kwargs)
        self.axis = axis
        self.inplace = kwargs.get('inplace', False)

    def call(self, inputs):
        return activations.softmax(
            inputs,
            axis=self.axis,
            inplace=self.inplace,
        )
