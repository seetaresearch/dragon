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


class Fundamental(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Fundamental, self).__init__(key, ctx, **kwargs)
        self.op_type = kwargs.get('op_type', 'Add')
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No arguments for fundamental ops."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': self.op_type,
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {}
        }

    def forward(self, x1, x2, y):
        inputs = [x1, x2]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output(x1.dtype)]
        return self.run(inputs, outputs)