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

from dragon.vm.torch.ops.modules.base import BaseModule


class Resize2d(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Resize2d, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', 'NNResize')
        self.dsize = kwargs.get('dsize', None)
        self.fx = kwargs.get('fx', None)
        self.fy = kwargs.get('fy', None)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': self.op_type,
            'arguments': {
                'fx': self.fx, 'fy': self.fy,
                'data_format': 'NCHW',
                'dsize_desc': [
                    '${{ANCHOR}}/dsize[{}]'.format(n)
                        for n in range(2)
                ],
            },
        }

    def update_arguments(self, A, dsize):
        if self.dsize:
            for i, e in enumerate(dsize):
                self.set_argument_i64('{}/dsize[{}]'.format(A, i), e)

    def forward(self, input, dsize=None):
        inputs = [input]; self.unify_devices(inputs)
        outputs = [self.register_output()]
        callback = lambda A: self.update_arguments(A, dsize)
        return self.run(inputs, outputs, callback=callback)


class RoIPool(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(RoIPool, self).__init__(key, dev, **kwargs)
        self.pool_h = kwargs.get('pooled_h', 0)
        self.pool_w = kwargs.get('pooled_w', 0)
        self.spatial_scale = kwargs.get('spatial_scale', 1.0)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'ROIPool',
            'arguments': {
                'pool_h': self.pool_h, 'pool_w': self.pool_w,
                'spatial_scale': self.spatial_scale,
            },
        }

    def forward(self, feature, rois, dsize=None):
        inputs = [feature, rois]; self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class RoIAlign(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(RoIAlign, self).__init__(key, dev, **kwargs)
        self.pool_h = kwargs.get('pooled_h', 0)
        self.pool_w = kwargs.get('pooled_w', 0)
        self.spatial_scale = kwargs.get('spatial_scale', 1.0)
        self.sampling_ratio = kwargs.get('sampling_ratio', 2)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'ROIAlign',
            'arguments': {
                'pool_h': self.pool_h, 'pool_w': self.pool_w,
                'spatial_scale': self.spatial_scale,
                'sampling_ratio': self.sampling_ratio,
            },
        }

    def forward(self, feature, rois, dsize=None):
        inputs = [feature, rois]; self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)