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


class DropBlock2d(Module):
    def __init__(self, block_size=7, kp=0.9,
                 alpha=1., decrement=0., inplace=False):
        super(DropBlock2d, self).__init__()
        self.kp = kp
        self.block_size = block_size
        self.alpha = alpha
        self.decrement = decrement
        self.inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'DropBlock2d',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'block_size': self.block_size,
                'keep_prob': self.kp,
                'alpha': self.alpha,
                'decrement': self.decrement,
                'data_format': 'NCHW',
                'phase': 'TRAIN',
            }
        }

    def forward(self, input):
        if not self.training: return input
        inputs = [input]
        self.unify_devices(inputs)
        outputs = [input if self.inplace else self.register_output(input.dtype)]
        return self.run(inputs, outputs)