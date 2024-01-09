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
"""Operator utilities."""

from dragon.core.framework import context
from dragon.core.framework import types
from dragon.vm.tensorflow.core.framework import constant_op


def convert_to_tensor(value, dtype=None, name=None):
    """Convert the given value to a tensor.

    Examples:

    ```python
    x = tf.convert_to_tensor([1, 2])
    y = tf.constant([1, 2])  # Equivalent
    ```

    Parameters
    ----------
    value : Union[number, Sequence, numpy.ndarray]
        The value to convert.
    dtype : str, optional
        The optional data type.
    name : str, optional
        The Optional name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `tf.constant(...)`_

    """
    if types.is_tensor(value):
        return value
    return constant_op.constant(value, dtype=dtype, name=name)


def name_scope(name):
    """Context-manager to nest the name as prefix for operations.

    Examples:

    ```python
    with tf.name_scope('my_scope'):
        x = tf.constant(1)
    print(x.name)
    ```

    Parameters
    ----------
    name : str
        The prefix name.

    Returns
    -------
    str
        The current nesting prefix.

    """
    return context.name_scope(name)


def device(device_name):
    """Context-manager to nest the device spec.

    Examples:

    ```python
    with tf.device('/gpu:0'):
        x = tf.constant(1)
    ```

    Parameters
    ----------
    device_name : str
        The device descriptor.

    Returns
    -------
    Dict
        The current default device spec.

    """
    if not isinstance(device_name, str):
        raise TypeError("The device function should be a str.")
    type_and_index = device_name.split("/")[-1]
    device_type, device_index = type_and_index.split(":")
    if device_type not in ["cpu", "gpu", "cuda", "mps"]:
        raise ValueError("Unsupported device type: " + device_type)
    try:
        device_index = int(device_index)
    except Exception:
        raise ValueError("The device index should be an integer.")
    return context.device(device_type, device_index)
