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


class OneHot(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(OneHot, self).__init__(key, ctx, **kwargs)
        self.depth = kwargs.get('depth', 1)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for concat op."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'OneHot',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'depth': self.depth,
            }
        }

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        return self.run(inputs, outputs)