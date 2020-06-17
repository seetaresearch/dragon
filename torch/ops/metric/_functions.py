# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.autograd import function


class Accuracy(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Accuracy, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.top_k = kwargs.get('top_k', 1)

    def attributes(self):
        return {
            'op_type': 'Accuracy',
            'arguments': {
                'axis': self.axis,
                'top_k': self.top_k,
            }
        }

    def forward(self, input, label):
        return self.dispatch([input, label], [self.alloc()])
