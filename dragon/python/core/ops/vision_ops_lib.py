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

from dragon.core.framework.ops import Operator


class _ConvNd(Operator):
    def __init__(self, key, dev, **kwargs):
        super(_ConvNd, self).__init__(key, dev, **kwargs)
        self.num_output = kwargs.get('dim_out', 1)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.dilations = kwargs.get('dilations', 1)
        self.group = kwargs.get('group', 1)
        self.padding = kwargs.get('padding', 'VALID')
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'kernel_shape': self.kernel_shape,
                'strides': self.strides,
                'pads': self.pads,
                'dilations': self.dilations,
                'padding': self.padding,
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class _PoolNd(Operator):
    def __init__(self, key, dev, **kwargs):
        super(_PoolNd, self).__init__(key, dev, **kwargs)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.padding = kwargs.get('padding', 'VALID')
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.mode = kwargs.get('mode', 'MAX')
        self.global_pooling = kwargs.get('global_pooling', False)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'kernel_shape': self.kernel_shape,
                'strides': self.strides,
                'pads': self.pads,
                'padding': self.padding,
                'ceil_mode': self.ceil_mode,
                'mode': self.mode,
                'data_format': self.data_format,
                'global_pooling': self.global_pooling,
            },
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class BiasAdd(Operator):
    def __init__(self, key, dev, **kwargs):
        super(BiasAdd, self).__init__(key, dev, **kwargs)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'BiasAdd',
            'arguments': {
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Conv2d(_ConvNd):
    def __init__(self, key, dev, **kwargs):
        super(Conv2d, self).__init__(key, dev, **kwargs)


class ConvTranspose2d(_ConvNd):
    def __init__(self, key, dev, **kwargs):
        super(ConvTranspose2d, self).__init__(key, dev, **kwargs)
        self.output_padding = kwargs.get('output_padding', None)
        self.output_shape = kwargs.get('output_shape', None)

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'kernel_shape': self.kernel_shape,
                'strides': self.strides,
                'pads': self.pads,
                'dilations': self.dilations,
                'padding': self.padding,
                'output_padding': self.output_padding,
                'output_shape': self.output_shape,
                'data_format': self.data_format,
            },
        }


class DepthToSpace(Operator):
    def __init__(self, key, dev, **kwargs):
        super(DepthToSpace, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', '2')
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'DepthToSpace',
            'arguments': {
                'block_size': self.block_size,
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class DepthwiseConv2d(_ConvNd):
    def __init__(self, key, dev, **kwargs):
        super(DepthwiseConv2d, self).__init__(key, dev, **kwargs)


class Pool2d(_PoolNd):
    def __init__(self, key, dev, **kwargs):
        super(Pool2d, self).__init__(key, dev, **kwargs)


class Resize(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Resize, self).__init__(key, dev, **kwargs)
        self.num_sizes = kwargs.get('num_sizes', 0)
        self.num_scales = kwargs.get('num_scales', 0)
        self.mode = kwargs.get('mode', 'NEAREST')
        self.align_corners = kwargs.get('align_corners', False)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'Resize',
            'arguments': {
                'mode': self.mode,
                'align_corners': self.align_corners,
                'sizes_descs': [
                    '${{HANDLE}}/sizes[{}]'
                    .format(n) for n in range(self.num_sizes)],
                'scales_descs': [
                    '${{HANDLE}}/scales[{}]'
                    .format(n) for n in range(self.num_scales)],
                'data_format': self.data_format,
            }
        }

    def feed(self, ws, handle, sizes, scales):
        for i in range(self.num_sizes):
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')
        for i in range(self.num_scales):
            self.feed_arg(
                ws, '{}/scales[{}]'.format(handle, i),
                scales[i], 'float32')

    def forward(self, inputs, sizes=None, scales=None):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, sizes, scales),
        )


class RoiAlign(Operator):
    def __init__(self, key, dev, **kwargs):
        super(RoiAlign, self).__init__(key, dev, **kwargs)
        self.pooled_h = kwargs.get('pooled_h', 0)
        self.pooled_w = kwargs.get('pooled_w', 0)
        self.spatial_scale = kwargs.get('spatial_scale', 1.0)
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

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class RoiPool(Operator):
    def __init__(self, key, dev, **kwargs):
        super(RoiPool, self).__init__(key, dev, **kwargs)
        self.pooled_h = kwargs.get('pooled_h', 7)
        self.pooled_w = kwargs.get('pooled_w', 7)
        self.spatial_scale = kwargs.get('spatial_scale', 1.)

    def attributes(self):
        return {
            'op_type': 'RoiPool',
            'arguments': {
                'pooled_h': self.pooled_h,
                'pooled_w': self.pooled_w,
                'spatial_scale': self.spatial_scale,
            },
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class SpaceToDepth(Operator):
    def __init__(self, key, dev, **kwargs):
        super(SpaceToDepth, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', '2')
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        return {
            'op_type': 'SpaceToDepth',
            'arguments': {
                'block_size': self.block_size,
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])
