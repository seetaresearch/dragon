# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.nn import Module


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.weight = weight
        # TODO(PhyscalX):  Dragon will support it later :).
        if weight is not None:
            raise NotImplementedError('WeightedLoss has been not implemented yet.')



class CrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SparseSoftmaxCrossEntropy' if self.reduce else 'SoftmaxCrossEntropy',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'axis': 1,
                'normalization': 'VALID' if self.size_average else 'NONE',
                'ignore_labels': () if self.ignore_index < 0 else (self.ignore_index),
            }
        }

    def forward(self, input, target):
        _assert_no_grad(target)
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)
