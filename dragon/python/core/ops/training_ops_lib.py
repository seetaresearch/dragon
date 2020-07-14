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
"""The training ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class ParamUpdate(Operator):
    def __init__(self, key, dev, **kwargs):
        super(ParamUpdate, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')
        self.op_handle = kwargs.get('op_handle', '')
        self.lr_mult = kwargs.get('lr_mult', 1)
        self.decay_mult = kwargs.get('decay_mult', 1)

    def attributes(self):
        return {
            'name': self.op_handle,
            'op_type': self.op_type,
            'arguments': {
                'lr_mult': float(self.lr_mult),
                'decay_mult': float(self.decay_mult),
            },
        }

    def forward(self, grad, param):
        return self.dispatch([grad], [param], no_grad=True)
