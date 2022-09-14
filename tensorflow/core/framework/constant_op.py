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

from dragon.core.ops import constant_ops


def constant(value, dtype=None, shape=None, name='Const'):
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
