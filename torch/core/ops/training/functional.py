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
"""Training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.ops.training import _functions


def accumulate_grad(grad, momentum=1):
    """Accumulate the gradient."""
    return _functions.GradAccumulate \
        .instantiate(grad.device, momentum=momentum).apply(grad)


def update_param(
    param,
    grad,
    op_type,
    op_handle,
    lr_mult=1,
    decay_mult=1,
):
    """Apply the parameter update."""
    return _functions.ParamUpdate \
        .instantiate(
            param.device,
            op_type=op_type,
            op_handle=op_handle,
            lr_mult=lr_mult,
            decay_mult=decay_mult,
        ).apply(param, grad)
