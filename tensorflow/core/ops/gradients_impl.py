# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Grad implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import grad_impl


def gradients(ys, xs, grad_ys=None, **kwargs):
    """Compute the symbolic derivatives of ``ys`` w.r.t. ``xs`` .

    By default, we will fill the gradient of ``ys`` with ones:

    ```python
    x = tf.ones(shape=(1,))
    y = x * 2
    dx = tf.gradients(y, x)  # [2,]
    ```

    You can set ``grad_ys`` to use an existing constant:

    ```python
    dy = tf.constant([2], dtype=tf.float32)
    dx = tf.gradients(y, x, dy)  # [4,]
    ```

    Do not call this method under eager execution:

    ```python
    # Wrong usage
    with dragon.eager_mode():
         x = tf.ones(shape=(1,))
         y = x * 2
         dx = tf.gradients(y, x)

    # Correct usage
    with dragon.eager_mode():
        x = tf.ones(shape=(1,))
        with tf.GradientTape() as tape:
            y = x * 2
        dx = tape.gradient(y, x)
    ```

    Parameters
    ----------
    ys : Sequence[dragon.Tensor]
        The target of derivatives.
    xs : Sequence[dragon.Tensor]
        The source with respect to the ``ys``.
    grad_ys : Sequence[dragon.Tensor], optional
        The input gradient for ``ys``.

    Returns
    -------
    Sequence[dragon.Tensor]
        The sum of derivatives for ``xs``.

    """
    return grad_impl.gradients(ys, xs, grad_ys)
