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
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight
        if weight is not None:
            raise NotImplementedError(
                'WeightedLoss has been not implemented yet.')


class NLLLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(NLLLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.reduction_v2 = {
            'elementwise_mean': 'VALID',
            'sum': 'SUM',
            'none': 'NONE'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'NLLLoss',
            'arguments': {
                'axis': 1,
                'reduction': self.reduction_v2,
                'ignore_labels': [] if self.ignore_index is None else [self.ignore_index],
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class BCEWithLogitsLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean',
        pos_weight=None,
    ):
        super(BCEWithLogitsLoss, self).__init__(
            weight, size_average, reduce, reduction)
        if pos_weight is not None:
            raise NotImplementedError(
                'Positive weight has been not implemented yet.')
        self.reduction_v2 = {
            'elementwise_mean': 'VALID',
            'sum': 'SUM',
            'none': 'NONE'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SigmoidCrossEntropy',
            'arguments': {
                'reduction': self.reduction_v2,
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class SCEWithLogitsLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean',
        pos_weight=None,
    ):
        super(SCEWithLogitsLoss, self).__init__(
            weight, size_average, reduce, reduction)
        if pos_weight is not None:
            raise NotImplementedError(
                'Positive weight has been not implemented yet.')
        self.reduction_v2 = {
            'elementwise_mean': 'VALID',
            'sum': 'SUM',
            'none': 'NONE'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SoftmaxCrossEntropy',
            'arguments': {
                'axis': 1,
                'reduction': self.reduction_v2,
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class CrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(CrossEntropyLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.reduction_v2 = {
            'elementwise_mean': 'VALID',
            'sum': 'SUM',
            'none': 'NONE'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SparseSoftmaxCrossEntropy',
            'arguments': {
                'axis': 1,
                'reduction': self.reduction_v2,
                'ignore_labels': [] if self.ignore_index is None else [self.ignore_index],
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class L1Loss(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(L1Loss, self).__init__(size_average, reduce, reduction)
        self.reduction_v2 = {
            'elementwise_mean': 'BATCH_SIZE',
            'sum': 'SUM'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'L1Loss',
            'arguments': {
                'reduction': self.reduction_v2,
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class MSELoss(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(MSELoss, self).__init__(size_average, reduce, reduction)
        self.reduction_v2 = {
            'elementwise_mean': 'BATCH_SIZE',
            'sum': 'SUM'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'L2Loss',
            'arguments': {
                'reduction': self.reduction_v2,
                'scale': 2., # We computes the 0.5 * (x - t) ** 2
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class SmoothL1Loss(_Loss):
    def __init__(
        self,
        size_average=None,
        beta=1.0,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)
        self.reduction_v2 = {
            'elementwise_mean': 'BATCH_SIZE',
            'sum': 'SUM'}[self.reduction]
        self.beta = beta
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SmoothL1Loss',
            'arguments': {
                'beta': self.beta,
                'reduction': self.reduction_v2,
            },
        }

    def forward(self, input, target, inside_w=None, outside_w=None):
        inputs = [input, target]
        if inside_w is not None: inputs += [inside_w]
        if outside_w is not None: inputs += [outside_w]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class SigmoidFocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        alpha=0.25,
        gamma=2.0,
        neg_id=0,
        ignore_index=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(SigmoidFocalLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.alpha, self.gamma, self.neg_id = alpha, gamma, neg_id
        self.ignore_index = ignore_index
        self.reduction_v2 = {
            'elementwise_mean': 'VALID',
            'sum': 'SUM',
            'none': 'NONE'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SigmoidFocalLoss',
            'arguments': {
                'axis': 1,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'neg_id': self.neg_id,
                'reduction': self.reduction_v2,
                'ignore_labels': [] if self.ignore_index is None else [self.ignore_index],
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class SoftmaxFocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        alpha=0.25,
        gamma=2.0,
        neg_id=0,
        ignore_index=None,
        reduce=None,
        reduction='elementwise_mean',
    ):
        super(SoftmaxFocalLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.alpha, self.gamma, self.neg_id = alpha, gamma, neg_id
        self.ignore_index = ignore_index
        self.reduction_v2 = {
            'elementwise_mean': 'VALID',
            'sum': 'SUM',
            'none': 'NONE'}[self.reduction]
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'SoftmaxFocalLoss',
            'arguments': {
                'axis': 1,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'neg_id': self.neg_id,
                'reduction': self.reduction_v2,
                'ignore_labels': [] if self.ignore_index is None else [self.ignore_index],
            },
        }

    def forward(self, input, target):
        inputs = [input, target]
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)