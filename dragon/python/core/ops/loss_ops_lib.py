# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Loss ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class _Loss(Operator):
    def __init__(self, key, dev, **kwargs):
        super(_Loss, self).__init__(key, dev, **kwargs)
        self.reduction = kwargs.get('reduction', 'MEAN')

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'reduction': self.reduction,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class L1Loss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(L1Loss, self).__init__(key, dev, **kwargs)


class L2Loss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(L2Loss, self).__init__(key, dev, **kwargs)


class NLLLoss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(NLLLoss, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.ignore_index = kwargs.get('ignore_index', None)

    def attributes(self):
        return {
            'op_type': 'NLLLoss',
            'arguments': {
                'axis': self.axis,
                'reduction': self.reduction,
                'ignore_index': self.ignore_index,
            }
        }


class SigmoidCrossEntropy(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SigmoidCrossEntropy, self).__init__(key, dev, **kwargs)


class SmoothL1Loss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SmoothL1Loss, self).__init__(key, dev, **kwargs)
        self.beta = kwargs.get('beta', 1.)

    def attributes(self):
        return {
            'op_type': 'SmoothL1Loss',
            'arguments': {
                'beta': float(self.beta),
                'reduction': self.reduction,
            }
        }


class SoftmaxCrossEntropy(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SoftmaxCrossEntropy, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)

    def attributes(self):
        return {
            'op_type': 'SoftmaxCrossEntropy',
            'arguments': {
                'axis': self.axis,
                'reduction': self.reduction,
            }
        }


class SparseSoftmaxCrossEntropy(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SparseSoftmaxCrossEntropy, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.ignore_index = kwargs.get('ignore_index', None)

    def attributes(self):
        return {
            'op_type': 'SparseSoftmaxCrossEntropy',
            'arguments': {
                'axis': self.axis,
                'reduction': self.reduction,
                'ignore_index': self.ignore_index,
            }
        }


class SigmoidFocalLoss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SigmoidFocalLoss, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.alpha = kwargs.get('alpha', 0.25)
        self.gamma = kwargs.get('gamma', 2.)
        self.negative_index = kwargs.get('negative_index', None)

    def attributes(self):
        return {
            'op_type': 'SigmoidFocalLoss',
            'arguments': {
                'axis': self.axis,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'negative_index': self.negative_index,
                'reduction': self.reduction,
            }
        }
