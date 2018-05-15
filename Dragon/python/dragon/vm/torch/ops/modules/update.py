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

from dragon.vm.torch.ops.modules.base import BaseModule


class Update(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Update, self).__init__(key, ctx, **kwargs)
        self.op_type = kwargs.get('op_type', 'Update')
        self.lr_mult = kwargs.get('lr_mult', 1.0)
        self.decay_mult = kwargs.get('decay_mult', 1.0)
        self.slot = kwargs.get('slot', '')
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No arguments for update ops."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': self.op_type,
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'lr_mult': self.lr_mult,
                'decay_mult': self.decay_mult,
                'slot': self.slot,
            }
        }

    def forward(self, param, grad):
        return self.run([grad], [param], auto_grad=False)