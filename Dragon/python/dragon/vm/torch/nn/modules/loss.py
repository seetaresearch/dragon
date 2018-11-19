# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
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
from dragon.vm.torch.nn.functional import _Reduction


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight
        if weight is not None:
            raise NotImplementedError('WeightedLoss has been not implemented yet.')


class NLLLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.normalization = {
            'elementwise_mean': 'VALID',
            'sum': 'None',
            'none': 'UNIT'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'NLLLoss',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'axis': 1,
                'normalization': self.normalization,
                'ignore_labels': () if self.ignore_index < 0 else (self.ignore_index),
            }
        }

    def forward(self, input, target):
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class BCEWithLogitsLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='elementwise_mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction)
        if pos_weight is not None:
            raise NotImplementedError('Positive weight has been not implemented yet.')
        self.normalization = {
            'elementwise_mean': 'VALID',
            'sum': 'None',
            'none': 'UNIT'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SigmoidCrossEntropy',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'normalization': self.normalization,
            }
        }

    def forward(self, input, target):
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class CrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.normalization = {
            'elementwise_mean': 'VALID',
            'sum': 'None',
            'none': 'UNIT'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SparseSoftmaxCrossEntropy',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'axis': 1,
                'normalization': self.normalization,
                'ignore_labels': () if self.ignore_index < 0 else (self.ignore_index),
            }
        }

    def forward(self, input, target):
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class L1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(L1Loss, self).__init__(size_average, reduce, reduction)
        self.normalization = {
            'elementwise_mean': 'BATCH_SIZE',
            'sum': 'None'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'L1Loss',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'normalization': self.normalization,
            }
        }

    def forward(self, input, target):
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class MSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)
        self.normalization = {
            'elementwise_mean': 'BATCH_SIZE',
            'sum': 'None'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'L2Loss',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'normalization': self.normalization,
            }
        }

    def forward(self, input, target):
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)


class SmoothL1Loss(_Loss):
    def __init__(self, size_average=None, beta=1.0,
                 reduce=None, reduction='elementwise_mean'):
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)
        self.normalization = {
            'elementwise_mean': 'BATCH_SIZE',
            'sum': 'None'}[self.reduction]
        self.beta = beta
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SmoothL1Loss',
            'n_inputs': 2, 'n_outputs': 1,
            'arguments': {
                'beta': self.beta,
                'normalization': self.normalization,
            }
        }

    def forward(self, input, target):
        inputs = [input, target]; self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype)]
        return self.run(inputs, outputs)
