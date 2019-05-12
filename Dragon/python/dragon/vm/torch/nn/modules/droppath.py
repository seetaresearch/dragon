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

from dragon.vm.torch.nn import Module


class DropPath(Module):
    def __init__(self, p=0.2, increment=0., inplace=False):
        super(DropPath, self).__init__()
        self.p = p
        self.increment = increment
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'DropPath',
            'arguments': {
                'prob': self.p,
                'increment': self.increment,
                'phase': 'TRAIN',
            }
        }

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)

    def forward(self, input):
        if not self.training: return input
        inputs = [input]
        self.unify_devices(inputs)
        outputs = [input if self.inplace else self.register_output()]
        return self.run(inputs, outputs)