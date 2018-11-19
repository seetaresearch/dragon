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

import dragon.core.mpi as mpi

from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.modules.update import Update, Collective


def _allreduce(grads):
    if not mpi.Is_Init(): return
    if not isinstance(grads, (list, tuple)): grads = [grads]
    ctx = MakeContext(inputs=grads)
    mode = mpi.GetParallelMode() + '_ALLREDUCE'
    key = 'torch/ops/collective/{}:{}/{}'.format(
        ctx[0].lower(), ctx[1], mode.lower())
    module = get_module(Collective, key, ctx, mode=mode)
    return module.forward(grads)


def _update(param, grad, op_type, slot,
            lr_mult=1.0, decay_mult=1.0):
    ctx = MakeContext(inputs=[param])
    key = 'torch/ops/{}/{}:{}/{}/{}'.format(op_type.lower(),
        ctx[0].lower(), ctx[1], slot, param.name)
    module = get_module(Update, key, ctx, op_type=op_type,
            lr_mult=lr_mult, decay_mult=decay_mult, slot=slot)
    return module.forward(param, grad)