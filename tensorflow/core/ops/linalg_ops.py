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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_ops.py>
#
# ------------------------------------------------------------
"""Linalg ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import constant_ops


def eye(num_rows, num_columns=None, dtype='float32', name=None):
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
