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

"""The implementation of the vision layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import normalization_ops
from dragon.core.ops import vision_ops
from dragon.vm.caffe.layer import Layer


class Convolution(Layer):
    r"""Apply the n-dimension convolution.

    The spatial output dimension is computed as:

    .. math::
        \begin{cases}
            \text{DK}_{size} = dilation *
                (\text{K}_{size} - 1) + 1 \\
            \text{Dim}_{out} = (\text{Dim}_{in} +
                2 * pad - \text{DK}_{size}) / stride + 1
        \end{cases}

    Examples:

    ```python
    layer {
        type: "Convolution"
        bottom: "input"
        top: "conv1"
        convolution_param {
            num_output: 32
            bias_term: true
            kernel_size: 3
            pad: 1
            stride: 1
            dilation: 1
            group: 1
            weight_filler {
                type: "xavier"
            }
            bias_filler {
                type: "constant"
                value: 0
            }
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Convolution, self).__init__(layer_param)
        param = layer_param.convolution_param
        self.arguments = {
            'out_channels': param.num_output,
            'kernel_shape': [int(e) for e in param.kernel_size],
            'strides': [int(e) for e in param.stride] if len(param.stride) > 0 else [1],
            'pads': [int(e) for e in param.pad] if len(param.pad) > 0 else [0],
            'dilations': [int(e) for e in param.dilation] if len(param.dilation) > 0 else [1],
            'group': int(param.group),
            'padding': 'VALID',
            'data_format': 'NCHW',
        }
        if param.HasField('kernel_h'):
            assert param.HasField('kernel_w')
            self.arguments['kernel_shape'] = [param.kernel_h, param.kernel_w]
        if param.HasField('stride_h'):
            assert param.HasField('stride_w')
            self.arguments['strides'] = [param.stride_h, param.stride_w]
        if param.HasField('pad_h'):
            assert param.HasField('pad_w')
            self.arguments['pads'] = [param.pad_h, param.pad_w]

        self.add_blob(filler=self.get_filler(param, 'weight_filler'))
        if param.bias_term:
            self.add_blob(filler=self.get_filler(param, 'bias_filler'))

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return vision_ops.conv2d(inputs, **self.arguments)


class Deconvolution(Convolution):
    r"""Apply the 2d deconvolution.

    The spatial output dimension is computed as:

    .. math::
        \begin{cases}
            \text{DK}_{size} = dilation *
                (\text{K}_{size} - 1) + 1 \\
            \text{Dim}_{out} = (\text{Dim}_{in} - 1) *
                stride + \text{DK}_{size} - 2 * pad
        \end{cases}

    Examples:

    ```python
    layer {
        type: "Deconvolution"
        bottom: "conv5"
        top: "conv5/upscale"
        convolution_param {
            num_output: 256
            bias_term: true
            kernel_size: 2
            pad: 0
            stride: 2
            dilation: 1
            group: 1
            weight_filler {
                type: "xavier"
            }
            bias_filler {
                type: "constant"
                value: 0
            }
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Deconvolution, self).__init__(layer_param)

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return vision_ops.conv2d_transpose(inputs, **self.arguments)


class DepthwiseConv2d(Layer):
    r"""Apply the 2d depthwise convolution.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

    The spatial output dimension is computed as:

    .. math::
        \begin{cases}
            \text{DK}_{size} = dilation *
                (\text{K}_{size} - 1) + 1 \\
            \text{Dim}_{out} = (\text{Dim}_{in} +
                2 * pad - \text{DK}_{size}) / stride + 1
        \end{cases}

    Examples:

    ```python
    layer {
        type: "DepthwiseConv2d"
        bottom: "input"
        top: "conv1"
        convolution_param {
            num_output: 32
            bias_term: true
            kernel_size: 3
            pad: 1
            stride: 1
            dilation: 1
            weight_filler {
                type: "xavier"
                variance_norm: FAN_OUT
            }
            bias_filler {
                type: "constant"
                value: 0
            }
        }
    }
    ```

    """
    def __init__(self, layer_param):
        super(DepthwiseConv2d, self).__init__(layer_param)
        param = layer_param.convolution_param
        self.arguments = {
            'out_channels': param.num_output,
            'kernel_shape': [int(e) for e in param.kernel_size],
            'strides': [int(e) for e in param.stride] if len(param.stride) > 0 else [1],
            'pads': [int(e) for e in param.pad] if len(param.pad) > 0 else [0],
            'padding': 'VALID',
            'data_format': 'NCHW',
        }
        if param.HasField('kernel_h'):
            assert param.HasField('kernel_w')
            self.arguments['kernel_shape'] = [param.kernel_h, param.kernel_w]
        if param.HasField('stride_h'):
            assert param.HasField('stride_w')
            self.arguments['strides'] = [param.stride_h, param.stride_w]
        if param.HasField('pad_h'):
            assert param.HasField('pad_w')
            self.arguments['pads'] = [param.pad_h, param.pad_w]

        self.add_blob(filler=self.get_filler(param, 'weight_filler'))
        if param.bias_term:
            self.add_blob(filler=self.get_filler(param, 'bias_filler'))

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return vision_ops.depthwise_conv2d(inputs, **self.arguments)


class LRN(Layer):
    r"""Apply the local response normalization.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    Examples:

    ```python
    layer {
        type: "LRN"
        bottom: "conv2"
        top: "conv2/norm"
        lrn_param {
            local_size: 5
            alpha: 1.
            beta: 0.75
            k: 1.
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(LRN, self).__init__(layer_param)
        param = layer_param.lrn_param
        if param.norm_region > 0:
            raise NotImplementedError('WITHIN_CHANNEL mode is not implemented.')
        self.arguments = {
            'size': param.local_size,
            'alpha': param.alpha,
            'beta': param.beta,
            'bias': param.k,
            'data_format': 'NCHW',
        }

    def __call__(self, bottom):
        return normalization_ops.local_response_norm(bottom, **self.arguments)


class Pooling(Layer):
    r"""Apply the n-dimension pooling.

    The spatial output dimension is computed as:

    .. math::
        \text{Dim}_{out} = (\text{Dim}_{in} +
            2 * pad - \text{K}_{size}) / stride + 1

    Examples:

    ```python
    layer {
        type: "Pooling"
        bottom: "conv2"
        top: "pool2"
        pooling_param {
            kernel_size: 3
            stride: 2
            pool: AVG
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Pooling, self).__init__(layer_param)
        param = layer_param.pooling_param
        self.arguments = {
            'data_format': 'NCHW',
            'mode': {0: 'MAX', 1: 'AVG'}[param.pool],
            'global_pooling': param.global_pooling,
        }
        if not param.HasField('kernel_h'):
            self.arguments['kernel_shape'] = [param.kernel_size]
        else:
            self.arguments['kernel_shape'] = [param.kernel_h, param.kernel_w]
        if not param.HasField('pad_h'):
            self.arguments['pads'] = [param.pad]
        else:
            self.arguments['pads'] = [param.pad_h, param.pad_w]
        if not param.HasField('stride_h'):
            self.arguments['strides'] = [param.stride]
        else:
            self.arguments['strides'] = [param.stride_h, param.stride_w]

    def __call__(self, bottom):
        return vision_ops.pool2d(bottom, **self.arguments)


class ROIAlign(Layer):
    r"""Apply the average roi align.
    `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.

    Examples:

    ```python
    layer {
        type: "ROIAlign"
        bottom: "conv5_3"
        top: "roi_pool4"
        roi_pooling_param {
            pooled_w: 7
            pooled_h: 7
            spatial_scale: 0.0625
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(ROIAlign, self).__init__(layer_param)
        param = layer_param.roi_pooling_param
        self.arguments = {
            'pool_h': int(param.pooled_h),
            'pool_w': int(param.pooled_w),
            'spatial_scale': param.spatial_scale,
        }

    def __call__(self, bottom):
        return vision_ops.roi_align(bottom, **self.arguments)


class ROIPooling(Layer):
    r"""Apply the max roi pooling.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    Examples:

    ```python
    layer {
        type: "ROIPooling"
        bottom: "conv5_3"
        top: "roi_pool4"
        roi_pooling_param {
            pooled_w: 7
            pooled_h: 7
            spatial_scale: 0.0625
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(ROIPooling, self).__init__(layer_param)
        param = layer_param.roi_pooling_param
        self.arguments = {
            'pool_h': int(param.pooled_h),
            'pool_w': int(param.pooled_w),
            'spatial_scale': param.spatial_scale,
        }

    def __call__(self, bottom):
        return vision_ops.roi_pool(bottom, **self.arguments)
