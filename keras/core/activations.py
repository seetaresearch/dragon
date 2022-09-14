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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import activation_ops
from dragon.core.ops import math_ops
from dragon.core.util import six
from dragon.vm.keras.core.utils import generic_utils


def elu(x, alpha=1., **kwargs):
    r"""Apply the exponential linear unit to input.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \text{ELU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.activations.elu(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=1.
        The value to :math:`\alpha`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.elu(x, alpha=alpha, **kwargs)


def exponential(x):
    r"""Apply the exponential activation to input.

    The **Exponential** function is defined as:

    .. math:: \text{Exp}(x) = \exp(x)

    Examples:

    ```python
    x = tf.constant([1, 2, 3], 'float32')
    print(tf.keras.activations.exponential(x))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.exp(x)


def hard_sigmoid(x, **kwargs):
    r"""Apply the hard sigmoid function to input.

    The **HardSigmoid** function is defined as:

    .. math:: \text{HardSigmoid}(x) = \max(0, \min(1, 0.2 * x + 0.5))

    Examples:

    ```python
    x = tf.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(tf.keras.activations.hard_sigmoid(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.hardsigmoid(x, **kwargs)


def linear(x):
    r"""Apply the linear activation to input.

    The **Linear** function is defined as:

    .. math:: \text{Linear}(x) = x

    Examples:

    ```python
    x = tf.constant([1, 2, 3], 'float32')
    print(tf.keras.activations.linear(x))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return x


def relu(x, alpha=0, max_value=None, **kwargs):
    r"""Apply the rectified linear unit to input.
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
    print(tf.keras.activations.relu(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    alpha : number, optional, default=0
        The value to :math:`\alpha`.
    max_value : number, optional
        The value to :math:`v_{max}`.

    """
    if max_value is not None:
        if alpha != 0:
            raise ValueError('Set either <alpha> or <max_value>.')
        if max_value != 6:
            raise ValueError('<max_value> can only be set to 6.')
        return activation_ops.relu6(x, **kwargs)
    return activation_ops.leaky_relu(x, alpha=alpha, **kwargs)


def selu(x, **kwargs):
    r"""Apply the scaled exponential linear unit to input.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    .. math::
        \text{SELU}(x) = 1.0507 *
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                1.67326 * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.activations.selu(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.selu(x, **kwargs)


def sigmoid(x, **kwargs):
    r"""Apply the sigmoid function to input.

    The **Sigmoid** function is defined as:

    .. math:: \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = tf.constant([0.2, 0.4, 0.6, 0.8, 1.0], 'float32')
    print(tf.keras.activations.sigmoid(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor

    """
    return activation_ops.sigmoid(x, **kwargs)


def softmax(x, axis=-1, **kwargs):
    r"""Apply the softmax function to input.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.keras.activations.softmax(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to reduce.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.softmax(x, axis=axis, **kwargs)


def swish(x):
    r"""Apply the swish function.
    `[Ramachandran et.al, 2017] <https://arxiv.org/abs/1710.05941>`_.

    The **Swish** function is defined as:

    .. math:: \text{Swish}(x) = x \cdot \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = tf.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(tf.keras.activations.swish(x))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.silu(x)


def tanh(x, **kwargs):
    r"""Apply the tanh function to input.

    The **Tanh** function is defined as:

    .. math:: \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}

    Examples:

    ```python
    x = tf.constant([0.2, 0.4, 0.6, 0.8, 1.0], 'float32')
    print(tf.keras.activations.tanh(x, inplace=False))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.tanh(x, **kwargs)


def get(identifier):
    """Return the activation function by identifier.

    Parameters
    ----------
    identifier : Union[callable, str]
        The identifier.

    Returns
    -------
    callable
        The activation function.

    """
    if identifier is None:
        return linear
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        return generic_utils.deserialize_keras_object(
            identifier, globals(), 'activation')
    else:
        raise TypeError(
            'Could not interpret the activation identifier: {}.'
            .format(identifier))
