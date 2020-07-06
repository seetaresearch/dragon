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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.tensor import TensorRef
from dragon.core.eager import context
from dragon.core.util import nest


class GradientInfo(object):
    """A class to store the known gradient relations."""

    def __init__(self, parent):
        self._parent = parent
        self._cost, self._wrt = [], []
        self._input = None

    @property
    def cost(self):
        return self._cost

    @property
    def input(self):
        return self._input

    @property
    def wrt(self):
        return self._wrt

    def add_cost(self, cost):
        self._cost.append(cost)

    def add_wrt(self, wrt):
        self._wrt.append(wrt)

    def make_pairs(self):
        return [(self._parent.id, wrt) for wrt in self._wrt]

    def required(self):
        return len(self._wrt) > 0

    def set_input(self, input):
        self._input = input


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

    # Record the gradient info (cost, wrt, input),
    # then, generate the gradient references once.
    for i, y in enumerate(ys):
        if y._grad is None:
            y._grad = GradientInfo(y)
        if grad_ys is not None:
            y._grad.set_input(grad_ys[i])
        for x in xs:
            if not hasattr(x, '_grad') or x._grad is None:
                x._grad = GradientInfo(x)
            y._grad.add_wrt(x.id)
            x._grad.add_cost(y)
            if i == 0:
                dxs.append(TensorRef(x.id + '_grad', x.shape, x.dtype))

    # Return the packed gradients.
    return dxs
