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
"""Input layer."""

from dragon.core.framework import workspace
from dragon.vm.tensorflow.core.framework import tensor_shape
from dragon.vm.tensorflow.core.ops import array_ops


def Input(shape=None, batch_size=None, name=None, dtype=None, tensor=None, **kwargs):
    """Create a symbolic tensor as the placeholder.

    Examples:

    ```python
    # Create a placeholder shape as (None, 8)
    x = tf.keras.Input(shape=(8,), dtype='float32')

    # Create a placeholder with determined ``batch_size``
    x = tf.keras.Input(shape=(8,), batch_size=8, dtype='float32')

    # Create a placeholder aliasing an existing tensor
    x = dragon.Tensor(shape=(8,), dtype='float32').constant()
    y = tf.keras.Input(tensor=x)
    ```

    Parameters
    ----------
    shape : Sequence[int], optional
        The input shape excluding ``batch_size``.
    batch_size : int, optional
        The dimension insert at the first axis.
    name : str, optional
        The optional placeholder name.
    dtype : str, optional
        The optional data type.
    tensor : dragon.Tensor, optional
        The existing tensor aliased to input.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if "batch_shape" in kwargs:
        batch_shape = kwargs.pop("batch_shape")
        if shape and batch_shape:
            raise ValueError("Specify <shape> or <batch_shape>, not both.")
        shape = batch_shape
    else:
        if shape is not None:
            if isinstance(shape, tensor_shape.TensorShape):
                shape = tuple(shape.as_list())
            elif isinstance(shape, int):
                shape = (shape,)
            shape = (batch_size,) + tuple(shape)
    if dtype is None:
        if tensor is not None:
            dtype = tensor.dtype
        else:
            dtype = "float32"
    if shape is None:
        if tensor is None:
            raise ValueError("Specify either <shape> or <tensor>.")
        else:
            shape = tensor.shape
    placeholder = array_ops.placeholder(dtype=dtype, shape=shape, name=name if name else "input")
    if tensor is not None:
        workspace.get_workspace().set_alias(tensor, placeholder.id)
    return placeholder
