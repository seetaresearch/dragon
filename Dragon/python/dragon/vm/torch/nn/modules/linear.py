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

import math

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.nn import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'InnerProduct',
            'n_inputs': 3 if self.bias else 2, 'n_outputs': 1,
            'arguments': {
                'num_output': self.weight.shape[0],
                'axis': -1,
            }
        }

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        inputs = [input, self.weight] + [self.bias] if self.bias else []
        self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)