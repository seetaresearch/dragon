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

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.nn import Module, Parameter
from dragon.vm.torch.ops.creation import zeros, ones
from dragon.vm.torch.module import RunOperator


class _GroupNorm(Module):
    def __init__(self, num_features, group=32,
                 eps=1e-5, affine=True):
        super(_GroupNorm, self).__init__()
        self.num_features = num_features
        self.group = group
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(Tensor(num_features))
            self.bias = Parameter(Tensor(num_features))
        else:
            self.weight = self.bias = None
        self.inputs = [self.weight, self.bias] if self.affine else []
        self.reset_parameters()
        self.register_op()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def register_op(self):
        self.op_meta = {
            'op_type': 'FusedGroupNorm' if self.affine else 'GroupNorm',
            'n_inputs': 3 if self.affine else 1, 'n_outputs': 1,
            'arguments': {
                'group': self.group,
                'axis': 1, # Data format: NCHW
                'eps': self.eps,
            }
        }

    def forward(self, input):
        inputs = [input] + self.inputs
        self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class GroupNorm1d(_GroupNorm):
    """Dragon does not use separate backend functions."""
    pass


class GroupNorm2d(_GroupNorm):
    """Dragon does not use separate backend functions."""
    pass


class GroupNorm3d(_GroupNorm):
    """Dragon does not use separate backend functions."""
    pass