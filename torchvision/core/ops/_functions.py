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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.autograd import function


class RoIPool(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(RoIPool, self).__init__(key, dev, **kwargs)
        self.pool_h = kwargs.get('pooled_h', 7)
        self.pool_w = kwargs.get('pooled_w', 7)
        self.spatial_scale = kwargs.get('spatial_scale', 1.)

    def attributes(self):
        return {
            'op_type': 'RoiPool',
            'arguments': {
                'pool_h': self.pool_h,
                'pool_w': self.pool_w,
                'spatial_scale': self.spatial_scale,
            },
        }

    def forward(self, input, boxes):
        return self.dispatch([input, boxes], [self.alloc()])


class RoIAlign(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(RoIAlign, self).__init__(key, dev, **kwargs)
        self.pooled_h = kwargs.get('pooled_h', 0)
        self.pooled_w = kwargs.get('pooled_w', 0)
        self.spatial_scale = kwargs.get('spatial_scale', 1.)
        self.sampling_ratio = kwargs.get('sampling_ratio', 2)

    def attributes(self):
        return {
            'op_type': 'RoiAlign',
            'arguments': {
                'pooled_h': self.pooled_h,
                'pooled_w': self.pooled_w,
                'spatial_scale': self.spatial_scale,
                'sampling_ratio': self.sampling_ratio,
            },
        }

    def forward(self, input, boxes):
        return self.dispatch([input, boxes], [self.alloc()])
