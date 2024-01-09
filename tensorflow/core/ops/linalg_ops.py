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
"""Linalg operators."""

from dragon.core.ops import constant_ops


def eye(num_rows, num_columns=None, dtype="float32", name=None):
    r"""Return a tensor constructed as the identity matrix.

    .. math:: \text{out} \leftarrow \text{diag}(1, 1, ..., 1)

    Matrix shape are determined by ``num_rows`` and ``num_columns``:

    ```python
    print(tf.eye(2))  # [[1., 0.], [0., 1.]]
    print(tf.eye(2, num_columns=3))  # [[1., 0., 0.], [0., 1., 0.]]
    ```

    Parameters
    ----------
    num_rows : Union[int, dragon.Tensor]
        The number output rows.
    num_columns : Union[int, dragon.Tensor], optional
        The number output cols.
    dtype : str, optional, default='float32'
        The optional data type.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = str(dtype) if dtype else dtype
    return constant_ops.eye(num_rows, num_columns, dtype=dtype, name=name)
