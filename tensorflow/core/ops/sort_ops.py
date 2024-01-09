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
"""Sort ops."""

from dragon.core.ops import sort_ops


def argsort(values, axis=-1, direction="ASCENDING", name=None):
    """Return the index of sorted elements along the given axis.

    By default, the last axis is chosen:

    ```python
    x = tf.constant([[1, 2, 3], [3, 2, 1]])
    index1 = tf.argsort(x)
    index2 = tf.argsort(x, axis=1)  # Equivalent
    ```

    Sort in the inverse order if ``direction`` is ``DESCENDING``:

    ```python
    x = tf.constant([1, 2, 3])
    index1 = tf.argsort(-x)
    index2 = tf.argsort(x, direction='DESCENDING')  # Equivalent
    ```

    Parameters
    ----------
    values : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to sort elements.
    direction : {'ASCENDING', 'DESCENDING'}, optional
        The sorting direction.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    descending = direction == "DESCENDING"
    value_and_index = sort_ops.sort(values, axis, descending, name=name)
    return value_and_index[1]


def sort(values, axis=-1, direction="ASCENDING", name=None):
    """Return the sorted elements along the given axis.

    By default, the last axis is chosen:

    ```python
    x = tf.constant([[1, 2, 3], [3, 2, 1]])
    value1 = tf.sort(x)
    value2 = tf.sort(x, axis=1)  # Equivalent
    ```

    Sort in the inverse order if ``direction`` is ``DESCENDING``:

    ```python
    x = tf.constant([1, 2, 3])
    value1 = -tf.sort(-x)
    value2 = tf.sort(x, direction='DESCENDING')  # Equivalent
    ```

    Parameters
    ----------
    values : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to sort elements.
    direction : {'ASCENDING', 'DESCENDING'}, optional
        The sorting direction.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    descending = direction == "DESCENDING"
    value_and_index = sort_ops.sort(values, axis, descending, name=name)
    return value_and_index[0]
