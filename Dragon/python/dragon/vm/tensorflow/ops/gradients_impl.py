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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.theano.gradient import grad as _Grad


def gradients(ys, xs, **kwargs):
    """Compute the gradients for variables with respect to the cost.

    Parameters
    ----------
    ys : sequence of Tensor
        The tensor(s) to be differentiated.
    xs : sequence of Tensor
        The tensor(s to be used for differentiation.

    Returns
    -------
    sequence of Tensor
        The gradients of variables.

    """
    dxs = []
    if not isinstance(ys, list): ys = [ys]
    for i, y in enumerate(ys):
        dy_wrt_dxs = _Grad(y, xs)
        if i == 0:
            if isinstance(dy_wrt_dxs, list):
                dxs.extend(dy_wrt_dxs)
            else:
                dxs.append(dy_wrt_dxs)
    return dxs