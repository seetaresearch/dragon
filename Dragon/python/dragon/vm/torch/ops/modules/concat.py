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


class Concat(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Concat, self).__init__(key, ctx, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for concat op."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'Concat',
            'n_inputs': 1, # Ignore
            'n_outputs': 1,
            'arguments': {
                'axis': self.axis,
            }
        }

    def forward(self, xs, y):
        inputs = xs; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output(xs[0].dtype)]
        return self.run(inputs, outputs)