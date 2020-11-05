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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/merge.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import array_ops
from dragon.core.ops import math_ops
from dragon.core.util import nest
from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer


class _Merge(Layer):
    """The base layer for elementwise merge functions."""

    def __init__(self, **kwargs):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
        """
        super(_Merge, self).__init__(**kwargs)
        self._reshape_required = False

    def _merge_function(self, inputs):
        """
        Merge the inputs.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        raise NotImplementedError

    def build(self, input_shape):
        """
        Connects the module into the graph.

        Args:
            self: (todo): write your description
            input_shape: (list): write your description
        """
        if not nest.is_sequence(input_shape):
            raise ValueError('Excepted a sequence of inputs for merge layer.')
        if len(input_shape) < 2:
            raise ValueError('Excepted at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')

    def call(self, inputs):
        """
        Call the sequence.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        if not nest.is_sequence(inputs):
            raise ValueError('Excepted a sequence of inputs for merge layer.')
        return self._merge_function(inputs)


class Add(_Merge):
    """The layer to add a sequence of inputs.

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.keras.layers.Add()([x, x, x]))
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``Add`` layer."""
        super(Add, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        """
        Merge a list of a list of - inputs.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        output = inputs[0]
        for i in range(1, len(inputs)):
            if output.id == inputs[i].id:
                output = output + inputs[i]
            else:
                output += inputs[i]
        return output


class Concatenate(_Merge):
    """The layer to concatenate a sequence of inputs.

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.keras.layers.Concatenate()([x, x, x]))
    ```

    """

    def __init__(self, axis=-1, **kwargs):
        """Create a ``Concatenate`` layer.

        Parameters
        ----------
        axis : int, optional, default=-1
            The axis to concatenate.

        """
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis

    def _merge_function(self, inputs):
        """
        Merge inputs together inputs.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        return array_ops.concat(inputs, self.axis)


class Maximum(_Merge):
    """The layer to compute the minimum of a sequence of inputs.

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.keras.layers.Maximum()([x, x, x]))
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``Maximum`` layer."""
        super(Maximum, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        """
        Merge the function.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.maximum([output, inputs[i]])
        return output


class Minimum(_Merge):
    """The layer to compute the minimum of a sequence of inputs.

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.keras.layers.Minimum()([x, x, x]))
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``Minimum`` layer."""
        super(Minimum, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        """
        Merge a list of inputs.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.minimum([output, inputs[i]])
        return output


class Multiply(_Merge):
    """The layer to multiply a sequence of inputs.

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.keras.layers.Multiply()([x, x, x]))
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``Multiply`` layer."""
        super(Multiply, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        """
        Merge the inputs of a list.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.mul([output, inputs[i]])
        return output


class Subtract(_Merge):
    """The layer to subtract two inputs.

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.keras.layers.Subtract()([x, x]))
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``Subtract`` layer."""
        super(Subtract, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Connects the module into the graph.

        Args:
            self: (todo): write your description
            input_shape: (list): write your description
        """
        super(Subtract, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('Exactly 2 inputs could subtract.')

    def _merge_function(self, inputs):
        """
        Merge input function.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        if len(inputs) != 2:
            raise ValueError('Exactly 2 inputs could subtract.')
        return inputs[0] - inputs[1]
