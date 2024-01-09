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
"""Constant operators."""

from dragon.core.ops import constant_ops


def constant(value, dtype=None, shape=None, name="Const"):
    """Return a tensor initialized from the value.

    Examples:

    ```python
    a = tf.constant(1)
    b = tf.constant(1, dtype='float32', shape=[1, 1, 1])
    c = tf.constant(numpy.ones((2, 3))
    ```

    Parameters
    ---------
    value : array_like
        The constant value.
    dtype : str, optional
        The optional data type.
    shape : Sequence[int], optional
        The optional tensor shape.
    name : str, optional
        The optional tensor name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = str(dtype) if dtype else dtype
    return constant_ops.constant(value, dtype=dtype, shape=shape, name=name)
