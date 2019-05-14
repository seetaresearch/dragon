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

from dragon.core import mpi as _mpi
from dragon.vm.torch.ops.modules.base import BaseModule


class Update(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Update, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', 'Update')
        self.lr_mult = kwargs.get('lr_mult', 1.0)
        self.decay_mult = kwargs.get('decay_mult', 1.0)
        self.slot = kwargs.get('slot', '')
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': self.op_type,
            'arguments': {
                'lr_mult': self.lr_mult,
                'decay_mult': self.decay_mult,
                'slot': self.slot,
            },
        }

    def forward(self, param, grad):
        self.unify_devices([param, grad])
        return self.run([grad], [param], auto_grad=False)


class Collective(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Collective, self).__init__(key, dev, **kwargs)
        self.mode = kwargs.get('mode', None)
        if self.mode is None:
            raise ValueError('Got invalid collective mode: {}'.format(self.mode))
        self.register_op()

    def register_op(self):
        idx, group = _mpi.AllowParallel()
        if idx == -1:
            raise RuntimeError(
                'The mpi node({}) dost not in groups.\n'
                'Set it using mpi.Parallel([..]).'.format(_mpi.Rank())
            )
        mpi_comm, mpi_group = _mpi.CreateGroup(root=group[0], incl=group)
        self.op_meta = {
            'op_type': 'CollectiveUpdate',
            'arguments': {
                'mode': self.mode,
                'comm': mpi_comm,
                'group': mpi_group,
                'root': group[0], # Assume the 1st node of group as root
            },
        }

    def forward(self, grads):
        self.unify_devices(grads)
        return self.run(grads, grads, auto_grad=False)


class Accumulate(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Accumulate, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Accumulate',
            'arguments': {
                'alpha': 1.,
                'beta': 1.,
            },
        }

    def forward(self, grads):
        self.unify_devices(grads)
        outputs = [grad.name + '[acc]' for grad in grads]
        return self.run(grads, outputs, auto_grad=False)