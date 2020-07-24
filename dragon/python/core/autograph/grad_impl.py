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
"""Gradient implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.tensor import TensorRef
from dragon.core.eager import context
from dragon.core.util import nest


class GradientInfo(object):
    """A class to store the known gradient relations."""

    def __init__(self, y, grad_y=None):
        self._y, self._grad_y, self._xs = y, grad_y, []

    @property
    def grad_y(self):
        return self._grad_y

    @property
    def xs(self):
        return self._xs

    @property
    def y(self):
        return self._y

    def add_x(self, x):
        self._xs.append(x)

    def required(self):
        return len(self._xs) > 0


def gradients(ys, xs, grad_ys=None):
    """Compute the symbolic derivatives of ``ys`` w.r.t. ``xs`` .

    By default, we will fill the gradient of ``ys`` with ones:

    ```python
    x = dragon.ones(shape=(1,))
    y = x * 2
    dx = dragon.gradients(y, x)  # [2,]
    ```

    You can set ``grad_ys`` to use an existing constant:

    ```python
    dy = dragon.constant([2], dtype=x.dtype)
    dx = dragon.gradients(y, x, dy)  # [4,]
    ```

    Do not call this method under eager execution:

    ```python
    # Wrong usage
    with dragon.eager_mode():
         x = dragon.ones(shape=(1,))
         y = x * 2
         dx = dragon.gradients(y, x)

    # Correct usage
    with dragon.eager_mode():
        x = dragon.ones(shape=(1,))
        with dragon.GradientTape() as tape:
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
    if context.executing_eagerly():
        raise RuntimeError('Use gradient tape under eager execution.')

    # Flatten the inputs.
    ys, xs, dxs = nest.flatten(ys), nest.flatten(xs), []
    if grad_ys is not None:
        grad_ys = nest.flatten(grad_ys)

    # Record the gradient info (y, grad_y, xs),
    # then, generate the gradient references once.
    for i, y in enumerate(ys):
        if y._grad is None:
            grad_y = grad_ys[i] if grad_ys is not None else None
            y._grad = GradientInfo(y, grad_y)
        for x in xs:
            y._grad.add_x(x)
            if i == 0:
                dxs.append(TensorRef(x.id + '_grad', x.shape, x.dtype))

    # Return the packed gradients.
    return dxs
