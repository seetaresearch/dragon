# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from dragon.core.tensor import Tensor
import dragon.vm.theano.tensor as T


def gradients(ys, xs, **kwargs):
    """Compute the gradients for variables with respect to the cost.

    Parameters
    ----------
    ys : Tensor or list of Tensor
        The tensor(s) to be differentiated.
    xs : Tensor or list of Tensor
        The tensor(s to be used for differentiation.

    Returns
    -------
    Tensor or list of Tensor
        The gradients of variables.

    """
    if not isinstance(ys, list):
        ys = [ys]
    for y in ys:
        dxs = T.grad(y, xs)
    return dxs