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
"""Vision ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class ConvNd(Operator):
    """ConvNd operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ConvNd, self).__init__(key, dev, **kwargs)
        self.num_output = kwargs.get('dim_out', 1)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.dilations = kwargs.get('dilations', 1)
        self.group = kwargs.get('group', 1)
        self.padding = kwargs.get('padding', 'VALID')
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
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
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class PoolNd(Operator):
    """PoolNd operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(PoolNd, self).__init__(key, dev, **kwargs)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.padding = kwargs.get('padding', 'VALID')
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.mode = kwargs.get('mode', 'MAX')
        self.global_pooling = kwargs.get('global_pooling', False)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        Returns a dictionary of attributes.

        Args:
            self: (todo): write your description
        """
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
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class BiasAdd(Operator):
    """BiasAdd operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(BiasAdd, self).__init__(key, dev, **kwargs)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        A dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'BiasAdd',
            'arguments': {'data_format': self.data_format},
        }

    def forward(self, inputs, inplace=False):
        """
        Forward computation

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Conv2d(ConvNd):
    """Conv2d operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Conv2d, self).__init__(key, dev, **kwargs)


class ConvTranspose2d(ConvNd):
    """ConvTranspose2d operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ConvTranspose2d, self).__init__(key, dev, **kwargs)
        self.output_padding = kwargs.get('output_padding', None)
        self.output_shape = kwargs.get('output_shape', None)

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
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
    """DepthToSpace operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(DepthToSpace, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', '2')
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        Return the attributes as a dict.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'DepthToSpace',
            'arguments': {
                'block_size': self.block_size,
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class DepthwiseConv2d(ConvNd):
    """DepthwiseConv2d operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(DepthwiseConv2d, self).__init__(key, dev, **kwargs)


class Pool2d(PoolNd):
    """Pool2d operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Pool2d, self).__init__(key, dev, **kwargs)


class Resize(Operator):
    """Resize operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Resize, self).__init__(key, dev, **kwargs)
        self.num_sizes = kwargs.get('num_sizes', 0)
        self.num_scales = kwargs.get('num_scales', 0)
        self.mode = kwargs.get('mode', 'NEAREST')
        self.align_corners = kwargs.get('align_corners', False)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        : return : class : numpy. array }

        Args:
            self: (todo): write your description
        """
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
        """
        Feed the number of the given handle.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            sizes: (int): write your description
            scales: (float): write your description
        """
        for i in range(self.num_sizes):
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')
        for i in range(self.num_scales):
            self.feed_arg(
                ws, '{}/scales[{}]'.format(handle, i),
                scales[i], 'float32')

    def forward(self, inputs, sizes=None, scales=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            sizes: (int): write your description
            scales: (todo): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, sizes, scales),
        )


class RoiAlign(Operator):
    """RoiAlign operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device attributes.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(RoiAlign, self).__init__(key, dev, **kwargs)
        self.pooled_h = kwargs.get('pooled_h', 0)
        self.pooled_w = kwargs.get('pooled_w', 0)
        self.spatial_scale = kwargs.get('spatial_scale', 1.0)
        self.sampling_ratio = kwargs.get('sampling_ratio', 2)

    def attributes(self):
        """
        A dictionary of the attributes for the sampler.

        Args:
            self: (todo): write your description
        """
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
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class RoiPool(Operator):
    """RoiPool operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a devi device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(RoiPool, self).__init__(key, dev, **kwargs)
        self.pooled_h = kwargs.get('pooled_h', 7)
        self.pooled_w = kwargs.get('pooled_w', 7)
        self.spatial_scale = kwargs.get('spatial_scale', 1.)

    def attributes(self):
        """
        A dict of the attributes for this object.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'RoiPool',
            'arguments': {
                'pooled_h': self.pooled_h,
                'pooled_w': self.pooled_w,
                'spatial_scale': self.spatial_scale,
            },
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class SpaceToDepth(Operator):
    """SpaceToDepth operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(SpaceToDepth, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', '2')
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        Return the attributes as a dict.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'SpaceToDepth',
            'arguments': {
                'block_size': self.block_size,
                'data_format': self.data_format,
            },
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])
