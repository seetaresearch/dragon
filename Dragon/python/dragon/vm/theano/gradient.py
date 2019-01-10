# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import dragon as dg


def grad(cost, wrt, **kwargs):
    """Compute the gradients for variables with respect to the cost.

    Parameters
    ----------
    cost : Tensor
        The cost.
    wrt : Tensor or list of Tensor
        The variables w.r.t the cost.

    Returns
    -------
    Tensor or list of Tensor
        The gradients of variables.

    Examples
    --------
    >>> import dragon as dg
    >>> x = dg.Tensor('x').Variable()
    >>> y = x * 2
    >>> dx = grad(y, x)

    >>> z = dg.Tensor('z').Variable()
    >>> y = x + z
    >>> dx, dz = grad(y, [x, z])

    """
    grads = []
    if not isinstance(wrt, list): wrt = [wrt]
    for w in wrt:
        cost.gradient.add_wrt(w.name)
        w.gradient.add_cost(cost)
        grads.append(dg.Tensor.Ref(
            name=w.name + '_grad',
                shape=w.shape, dtype=w.dtype))
    if len(grads) == 1: return grads[0]
    return grads


def disconnected_grad(x):
    """Return the identity of input with truncated gradient flow.

    The expression itself is unaffected, but the gradient is stopped.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The identity of input.

    """
    return dg.ops.StopGradient(x)
