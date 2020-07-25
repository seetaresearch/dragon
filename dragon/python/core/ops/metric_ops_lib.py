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
"""Metric ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Metric(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Metric, self).__init__(key, dev, **kwargs)
        self.reduction = kwargs.get('reduction', 'MEAN')

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Accuracy(Metric):
    def __init__(self, key, dev, **kwargs):
        super(Accuracy, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.top_k = kwargs.get('top_k', 1)
        self.ignore_index = kwargs.get('ignore_index', None)

    def attributes(self):
        return {
            'op_type': 'Accuracy',
            'arguments': {
                'axis': self.axis,
                'top_k': self.top_k,
                'ignore_index': self.ignore_index,
            }
        }
