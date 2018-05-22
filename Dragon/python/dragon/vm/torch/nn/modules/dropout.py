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


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Dropout',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'prob': self.p,
            }
        }

    def forward(self, input):
        if not input.requires_grad: return input
        inputs = [input]
        self.unify_devices(inputs)
        outputs = [input if self.inplace else self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class Dropout2d(Dropout):
    """Dragon does not use separate backend functions."""
    pass


class Dropout3d(Dropout):
    """Dragon does not use separate backend functions."""
    pass