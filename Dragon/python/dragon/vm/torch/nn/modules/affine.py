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
from dragon.vm.torch.ops.builtin import zeros, ones


class Affine(Module):
    def __init__(
        self,
        num_features,
        bias=True,
        fix_weight=False,
        fix_bias=False,
        inplace=False,
    ):
        super(Affine, self).__init__()
        self.num_features = num_features
        self.inplace = inplace
        if not fix_weight:
            self.weight = Parameter(ones(num_features))
            if inplace:
                raise ValueError('Inplace computation requires fixed weight.')
        else:
            self.register_buffer('weight', ones(num_features))
        if bias:
            if not fix_bias:
                self.bias = Parameter(zeros(num_features))
            else:
                self.register_buffer('bias', zeros(num_features))
        else:
            self.bias = None
        self.inputs = [self.weight, self.bias] if bias else [self.weight]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Affine',
            'arguments': {
                'axis': 1, # Data format: NCHW
                'num_axes': 1,
            }
        }

    def extra_repr(self):
        s = '{num_features}, inplace={inplace}'.format(**self.__dict__)
        if self.bias is None: s += ', bias=False'
        return s

    def forward(self, input):
        inputs = [input] + self.inputs
        self.unify_devices(inputs)
        outputs = [input if self.inplace else self.register_output()]
        return self.run(inputs, outputs)