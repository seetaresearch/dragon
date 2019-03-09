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

"""The Implementation of the vision layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
from ..layer import Layer


class ConvolutionLayer(Layer):
    """The implementation of ``ConvolutionLayer``.

    Parameters
    ----------
    num_output : int
        The output channels. Refer `ConvolutionParameter.num_output`_.
    bias_term : boolean
        Whether to use bias. Refer `ConvolutionParameter.bias_term`_.
    pad : list of int
        The zero padding size(s). Refer `ConvolutionParameter.pad`_.
    kernel_size : list of int
        The kernel size(s). Refer `ConvolutionParameter.kernel_size`_.
    stride : list of int
        The stride(s). Refer `ConvolutionParameter.stride`_.
    dilation : list of int
        The dilation(s). Refer `ConvolutionParameter.dilation`_.
    group : int
         The group size. Refer `ConvolutionParameter.group`_.
    weight_filler : FillerParameter
        The filler of weights. Refer `ConvolutionParameter.weight_filler`_.
    bias_filler : FillerParameters
        The filler of bias. Refer `ConvolutionParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(ConvolutionLayer, self).__init__(LayerParameter)
        param = LayerParameter.convolution_param
        self.arguments = {
            'num_output': param.num_output,
            'kernel_shape': [int(element) for element in param.kernel_size],
            'strides': [int(element) for element in param.stride] if len(param.stride) > 0 else [1],
            'pads': [int(element) for element in param.pad] if len(param.pad) > 0 else [0],
            'dilations': [int(element) for element in param.dilation] if len(param.dilation) > 0 else [1],
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

        # Add weights and biases
        self.AddBlob(filler=self.GetFiller(param, 'weight_filler'))
        if param.bias_term:
            self.AddBlob(filler=self.GetFiller(param, 'bias_filler'))

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return dragon.ops.Conv2d(inputs, **self.arguments)


class DepthwiseConvolutionLayer(Layer):
    """The implementation of ``DepthwiseConvolutionLayer``.

    Parameters
    ----------
    num_output : int
        The output channels. Refer `ConvolutionParameter.num_output`_.
    bias_term : boolean
        Whether to use bias. Refer `ConvolutionParameter.bias_term`_.
    pad : list of int
        The zero padding size(s). Refer `ConvolutionParameter.pad`_.
    kernel_size : list of int
        The kernel size(s). Refer `ConvolutionParameter.kernel_size`_.
    stride : list of int
        The stride(s). Refer `ConvolutionParameter.stride`_.
    weight_filler : FillerParameter
        The filler of weights. Refer `ConvolutionParameter.weight_filler`_.
    bias_filler : FillerParameters
        The filler of bias. Refer `ConvolutionParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(DepthwiseConvolutionLayer, self).__init__(LayerParameter)
        param = LayerParameter.convolution_param
        self.arguments = {
            'num_output': param.num_output,
            'kernel_shape': [int(element) for element in param.kernel_size],
            'strides': [int(element) for element in param.stride] if len(param.stride) > 0 else [1],
            'pads': [int(element) for element in param.pad] if len(param.pad) > 0 else [0],
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

        # Add weights and biases
        self.AddBlob(filler=self.GetFiller(param, 'weight_filler'))
        if param.bias_term:
            self.AddBlob(filler=self.GetFiller(param, 'bias_filler'))

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return dragon.ops.DepthwiseConv2d(inputs, **self.arguments)


class DeconvolutionLayer(ConvolutionLayer):
    """The implementation of ``DeconvolutionLayer``.

    Parameters
    ----------
    num_output : int
        The output channels. Refer `ConvolutionParameter.num_output`_.
    bias_term : boolean
        Whether to use bias. Refer `ConvolutionParameter.bias_term`_.
    pad : list of int
        The zero padding size(s). Refer `ConvolutionParameter.pad`_.
    kernel_size : list of int
        The kernel size(s). Refer `ConvolutionParameter.kernel_size`_.
    stride : list of int
        The stride(s). Refer `ConvolutionParameter.stride`_.
    dilation : list of int
        The dilation(s). Refer `ConvolutionParameter.dilation`_.
    group : int
         The group size. Refer `ConvolutionParameter.group`_.
    weight_filler : FillerParameter
        The filler of weights. Refer `ConvolutionParameter.weight_filler`_.
    bias_filler : FillerParameters
        The filler of bias. Refer `ConvolutionParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(DeconvolutionLayer, self).__init__(LayerParameter)

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return dragon.ops.ConvTranspose2d(inputs, **self.arguments)


class PoolingLayer(Layer):
    """The implementation of ``PoolingLayer``.

    Parameters
    ----------
    pool : PoolMethod
        The method. Refer `PoolingParameter.pool`_.
    pad : list of int
        The zero padding size(s). Refer `PoolingParameter.pad`_.
    pad_h : int
        The padding size of height. Refer `PoolingParameter.pad_h`_.
    pad_w : int
        The padding size of width. Refer `PoolingParameter.pad_w`_.
    kernel_size : list of int
        The kernel size(s). Refer `PoolingParameter.kernel_size`_.
    kernel_h : int
        The kernel size of height. Refer `PoolingParameter.kernel_h`_.
    kernel_w : int
        The kernel size of width. Refer `PoolingParameter.kernel_w`_.
    stride : list of int
        The strides. Refer `PoolingParameter.stride`_.
    stride_h : int
        The stride of height. Refer `PoolingParameter.stride_h`_.
    stride_w : int
        The stride of width. Refer `PoolingParameter.stride_w`_.

    """
    def __init__(self, LayerParameter):
        super(PoolingLayer, self).__init__(LayerParameter)
        param = LayerParameter.pooling_param
        self.arguments = {
            'mode': {0: 'MAX', 1: 'AVG'}[param.pool],
            'data_format': 'NCHW',
            'global_pooling': param.global_pooling,
            'ceil_mode': True,
        }
        if not param.HasField('kernel_h'): self.arguments['kernel_shape'] = [param.kernel_size]
        else: self.arguments['kernel_shape'] = [param.kernel_h, param.kernel_w]

        if not param.HasField('pad_h'): self.arguments['pads'] = [param.pad]
        else: self.arguments['pads'] = [param.pad_h, param.pad_w]

        if not param.HasField('stride_h'): self.arguments['strides'] = [param.stride]
        else: self.arguments['strides'] = [param.stride_h, param.stride_w]

    def LayerSetup(self, bottom):
        return dragon.ops.Pool2d(bottom, **self.arguments)


class ROIPoolingLayer(Layer):
    """The implementation of ``ROIPoolingLayer``.

    Parameters
    ----------
    pooled_h : int
        The height of pooled output. Refer `ROIPoolingParameter.pooled_h`_.
    pooled_w : int
        The width of pooled output. Refer `ROIPoolingParameter.pooled_w`_.
    spatial_scale : float
         The ``inverse`` of down-sampling multiples. Refer `ROIPoolingParameter.spatial_scale`_.

    """
    def __init__(self, LayerParameter):
        super(ROIPoolingLayer, self).__init__(LayerParameter)
        param = LayerParameter.roi_pooling_param
        self.arguments = {
            'pool_h': int(param.pooled_h),
            'pool_w': int(param.pooled_w),
            'spatial_scale': param.spatial_scale,
        }

    def LayerSetup(self, bottom):
        return dragon.ops.ROIPool(bottom, **self.arguments)


class ROIAlignLayer(Layer):
    """The implementation of ``ROIAlignLayer``.

    Parameters
    ----------
    pooled_h : int
        The height of pooled output. Refer `ROIPoolingParameter.pooled_h`_.
    pooled_w : int
        The width of pooled output. Refer `ROIPoolingParameter.pooled_w`_.
    spatial_scale : float
         The ``inverse`` of down-sampling multiples. Refer `ROIPoolingParameter.spatial_scale`_.

    """
    def __init__(self, LayerParameter):
        super(ROIAlignLayer, self).__init__(LayerParameter)
        param = LayerParameter.roi_pooling_param
        self.arguments = {
            'pool_h': int(param.pooled_h),
            'pool_w': int(param.pooled_w),
            'spatial_scale': param.spatial_scale,
        }

    def LayerSetup(self, bottom):
        return dragon.ops.ROIAlign(bottom, **self.arguments)


class LRNLayer(Layer):
    """The implementation of ``LRNLayer``.

    Parameters
    ----------
    local_size : int
        Refer `LRNParameter.local_size`_.
    alpha : float
        Refer `LRNParameter.alpha`_.
    beta : float
        Refer `LRNParameter.beta`_.
    norm_region : NormRegion
        Refer `LRNParameter.norm_region`_.
    k : float
        Refer `LRNParameter.k`_.

    """
    def __init__(self, LayerParameter):
        super(LRNLayer, self).__init__(LayerParameter)
        param = LayerParameter.lrn_param
        self.arguments = {
            'local_size': param.local_size,
            'alpha': param.alpha,
            'beta': param.beta,
            'mode': {0: 'ACROSS_CHANNELS', 1: 'WITHIN_CHANNEL'}[param.norm_region],
            'data_format': 'NCHW',
        }

    def LayerSetup(self, bottom):
        return dragon.ops.LRN(bottom, **self.arguments)


class NNResizeLayer(Layer):
    """The implementation of ``NNResizeLayer``.

    Parameters
    ----------
    shape : caffe_pb2.BlobShape
        The output shape. Refer `ResizeParameter.shape`_.
    fx : float
        The scale factor of height. Refer `ResizeParameter.fx`_.
    fy : float
        The scale factor of width. Refer `ResizeParameter.fy`_.

    """
    def __init__(self, LayerParameter):
        super(NNResizeLayer, self).__init__(LayerParameter)
        param = LayerParameter.resize_param
        dsize = [int(dim) for dim in param.shape.dim] \
            if param.HasField('shape') else None
        self.arguments = {
            'dsize': dsize,
            'fx': float(param.fx),
            'fy': float(param.fy),
            'data_format': 'NCHW',
        }

    def LayerSetup(self, bottom):
        if self.arguments['dsize'] is None:
            if not isinstance(bottom, (tuple, list)) or len(bottom) != 2:
                raise ValueError('The second bottom should be provided to determine the shape.')
            self.arguments['shape_like'] = bottom[1]
            bottom = bottom[0]
        return dragon.ops.NNResize(bottom, **self.arguments)


class BilinearResizeLayer(Layer):
    """The implementation of ``BilinearResizeLayer``.

    Parameters
    ----------
    shape : caffe_pb2.BlobShape
        The output shape. Refer `ResizeParameter.shape`_.
    fx : float
        The scale factor of height. Refer `ResizeParameter.fx`_.
    fy : float
        The scale factor of width. Refer `ResizeParameter.fy`_.

    """
    def __init__(self, LayerParameter):
        super(BilinearResizeLayer, self).__init__(LayerParameter)
        param = LayerParameter.resize_param
        dsize = [int(dim) for dim in param.shape.dim] \
            if param.HasField('shape') else None
        self.arguments = {
            'dsize': dsize,
            'fx': float(param.fx),
            'fy': float(param.fy),
            'data_format': 'NCHW',
        }

    def LayerSetup(self, bottom):
        if self.arguments['dsize'] is None:
            if not isinstance(bottom, (tuple, list)) or len(bottom) != 2:
                raise ValueError('The second bottom should be provided to determine the shape.')
            self.arguments['shape_like'] = bottom[1]
            bottom = bottom[0]
        return dragon.ops.BilinearResize(bottom, **self.arguments)


class DropBlockLayer(Layer):
    """The implementation of ``DropBlock2dLayer``.

    Parameters
    ----------
    block_size : int
        The size of dropping block. Refer ``DropBlockParameter.block_size``.
    keep_prob : float
        The prob of keeping. Refer ``DropBlockParameter.keep_prob``.
    alpha : float
        The scale factor to gamma. Refer ``DropBlockParameter.alpha``.
    decrement : float
        The decrement to keep prob. Refer ``DropBlockParameter.decrement``.

    """
    def __init__(self, LayerParameter):
        super(DropBlockLayer, self).__init__(LayerParameter)
        param = LayerParameter.drop_block_param
        self.arguments = {
            'block_size': param.block_size,
            'keep_prob': param.keep_prob,
            'alpha': param.alpha,
            'decrement': param.decrement,
            'data_format': 'NCHW',
        }

    def LayerSetup(self, bottom):
        return dragon.ops.DropBlock2d(bottom, **self.arguments)