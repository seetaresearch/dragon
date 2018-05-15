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


class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self._inplace = inplace
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Relu',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {}
        }

    def forward(self, x):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [x if self._inplace else self.register_output(x.dtype)]
        return self.run(inputs, outputs)