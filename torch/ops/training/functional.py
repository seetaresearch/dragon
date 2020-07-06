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

from dragon.core.util import nest
from dragon.vm.torch.ops.training import _functions


def grad_accumulate(grads):
    """Accumulate the gradients."""
    grads = nest.flatten(grads)
    if len(grads) == 0:
        return
    return _functions.GradAccumulate \
        .instantiate(grads[0].device).apply(grads)


def param_update(
    param,
    grad,
    op_type,
    op_handle,
    lr_mult=1,
    decay_mult=1,
):
    """Apply the param update."""
    return _functions.ParamUpdate \
        .instantiate(
            param.device,
            op_type=op_type,
            op_handle=op_handle,
            lr_mult=lr_mult,
            decay_mult=decay_mult,
        ).apply(param, grad)
