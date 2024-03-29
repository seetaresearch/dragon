# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Merging layers."""

from dragon.core.ops import array_ops
from dragon.core.ops import math_ops
from dragon.core.util import nest
from dragon.vm.keras.core.engine.base_layer import Layer


class _Merge(Layer):
    """Base layer to call elementwise merge function."""

    def __init__(self, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self._reshape_required = False

    def _merge_function(self, inputs):
        """Merge function."""

    def build(self, input_shape):
        if not nest.is_sequence(input_shape[0]):
            raise ValueError("Excepted a sequence of inputs for merge layer.")
        if len(input_shape) < 2:
            raise ValueError(
                "Excepted at least 2 inputs. " "Got " + str(len(input_shape)) + " inputs."
            )

    def call(self, inputs):
        if not nest.is_sequence(inputs):
            raise ValueError("Excepted a sequence of inputs for merge layer.")
        return self._merge_function(inputs)


class Add(_Merge):
    """Add a sequence of inputs.

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
        output = inputs[0]
        for i in range(1, len(inputs)):
            if output.id == inputs[i].id:
                output = output + inputs[i]
            else:
                output += inputs[i]
        return output


class Concatenate(_Merge):
    """Concatenate a sequence of inputs.

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
        return array_ops.concat(inputs, self.axis)


class Maximum(_Merge):
    """Compute the minimum of a sequence of inputs.

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
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.maximum([output, inputs[i]])
        return output


class Minimum(_Merge):
    """Compute the minimum of a sequence of inputs.

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
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.minimum([output, inputs[i]])
        return output


class Multiply(_Merge):
    """Multiply a sequence of inputs.

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
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = math_ops.mul([output, inputs[i]])
        return output


class Subtract(_Merge):
    """Subtract two inputs.

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
        super(Subtract, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError("Exactly 2 inputs could subtract.")

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError("Exactly 2 inputs could subtract.")
        return inputs[0] - inputs[1]
