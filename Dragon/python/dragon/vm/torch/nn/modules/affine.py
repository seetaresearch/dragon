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

from dragon.vm.torch.nn import Module, Parameter
from dragon.vm.torch.ops.creation import zeros, ones


class Affine(Module):
    def __init__(self, num_features, bias=True, fix_weight=False, fix_bias=False):
        super(Affine, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(ones(num_features), requires_grad=not fix_weight)
        if bias:
            self.bias = Parameter(zeros(num_features), requires_grad=not fix_bias)
        else:
            self.bias = None
        self.inputs = [self.weight, self.bias] if bias else [self.weight]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Affine',
            'n_inputs': 3 if self.bias else 2, 'n_outputs': 1,
            'arguments': {
                'axis': 1, # Data format: NCHW
                'num_axes': 1,
            }
        }

    def forward(self, input):
        inputs = [input] + self.inputs
        self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)