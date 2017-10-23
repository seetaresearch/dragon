# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import dragon.ops as ops

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
    >>> x = Tensor('x').Variable()
    >>> y = x * 2
    >>> dx = grad(y, x)

    >>> z = Tensor('z').Variable()
    >>> y = x + z
    >>> dx, dz = grad(y, [x, z])

    """
    grads = []
    if not isinstance(wrt, list): wrt = [wrt]
    for w in wrt:
        cost.grad_wrts.append(w.name)
        w.grad_objs.append(cost.name)
        grads.append(Tensor(w.name + '_grad'))
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
    return ops.StopGradient(x)
