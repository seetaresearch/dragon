# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon
from dragon.core.tensor import Tensor
from layer import Layer
import dragon.ops as ops

class ConvolutionLayer(Layer):
    def __init__(self, LayerParameter):
        super(ConvolutionLayer, self).__init__(LayerParameter)
        param = LayerParameter.convolution_param
        self._param = {'num_output': param.num_output,
                       'kernel_size': [int(element) for element in param.kernel_size],
                       'stride': [int(element) for element in param.stride] if len(param.stride) > 0 else [1],
                       'pad': [int(element) for element in param.pad] if len(param.pad) > 0 else [0],
                       'dilation': [int(element) for element in param.dilation] if len(param.dilation) > 0 else [1],
                       'group': int(param.group)}
        if param.HasField('kernel_h'):
            assert param.HasField('kernel_w')
            self._param['kernel'] = [param.kernel_h, param.kernel_w]
        if param.HasField('stride_h'):
            assert param.HasField('stride_w')
            self._param['stride'] = [param.stride_h, param.stride_w]
        if param.HasField('pad_h'):
            assert param.HasField('pad_w')
            self._param['pad'] = [param.pad_h, param.pad_w]
        weight = Tensor(LayerParameter.name + '@param0')
        weight_diff = Tensor(LayerParameter.name + '@param0_grad')
        if len(LayerParameter.param) > 0:
            if LayerParameter.param[0].lr_mult <= 0: weight_diff = None
        self.Fill(weight, param, 'weight_filler')
        self._blobs.append({'data': weight, 'diff': weight_diff})

        if param.bias_term:
            bias = Tensor(LayerParameter.name + '@param1')
            bias_diff = Tensor(LayerParameter.name + '@param1_grad')
            self.Fill(bias, param, 'bias_filler')
            if len(LayerParameter.param) > 1:
                if LayerParameter.param[1].lr_mult <= 0: bias_diff = None
            self._blobs.append({'data': bias, 'diff': bias_diff})

    def Setup(self, bottom):
        super(ConvolutionLayer, self).Setup(bottom)
        return ops.Conv2D(bottom + [blob['data'] for blob in self._blobs], **self._param)


class DeconvolutionLayer(ConvolutionLayer):
    def __init__(self, LayerParameter):
        super(DeconvolutionLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(DeconvolutionLayer, self).Setup(bottom)
        return ops.Deconv2D(bottom + [blob['data'] for blob in self._blobs], **self._param)


class PoolingLayer(Layer):
    def __init__(self, LayerParameter):
        super(PoolingLayer, self).__init__(LayerParameter)
        param = LayerParameter.pooling_param
        self._param = {'kernel_size': [param.kernel_size],
                       'stride': [param.stride],
                       'pad': [param.pad],
                       'mode': {0: 'MAX_POOLING', 1: 'AVG_POOLING'}[param.pool]}

    def Setup(self, bottom):
        input = bottom[0] if isinstance(bottom, list) else bottom
        super(PoolingLayer, self).Setup(bottom)
        return ops.Pool2D(input, **self._param)


class ROIPoolingLayer(Layer):
    def __init__(self, LayerParameter):
        super(ROIPoolingLayer, self).__init__(LayerParameter)
        param = LayerParameter.roi_pooling_param
        self._param = {'pool_h': int(param.pooled_h),
                       'pool_w': int(param.pooled_w),
                       'spatial_scale': param.spatial_scale}

    def Setup(self, bottom):
        super(ROIPoolingLayer, self).Setup(bottom)
        return ops.ROIPooling(bottom, **self._param)


class ROIAlignLayer(Layer):
    def __init__(self, LayerParameter):
        super(ROIAlignLayer, self).__init__(LayerParameter)
        param = LayerParameter.roi_pooling_param
        self._param = {'pool_h': int(param.pooled_h),
                       'pool_w': int(param.pooled_w),
                       'spatial_scale': param.spatial_scale}

    def Setup(self, bottom):
        super(ROIAlignLayer, self).Setup(bottom)
        return ops.ROIAlign(bottom, **self._param)


class LRNLayer(Layer):
    def __init__(self, LayerParameter):
        super(LRNLayer, self).__init__(LayerParameter)
        param = LayerParameter.lrn_param
        self._param = {'local_size': param.local_size,
                       'alpha': param.alpha,
                       'beta': param.beta,
                       'mode': {0: 'ACROSS_CHANNELS', 1: 'WITHIN_CHANNEL'}[param.norm_region]}
    def Setup(self, bottom):
        super(LRNLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.LRN(input, **self._param)


class NNResizeLayer(Layer):
    def __init__(self, LayerParameter):
        super(NNResizeLayer, self).__init__(LayerParameter)
        param = LayerParameter.nnresize_param
        dsize = [int(dim) for dim in param.shape.dim] \
            if param.HasField('shape') else []
        self._param = {'dsize': dsize,
                       'fx': float(param.fx),
                       'fy': float(param.fy)}

    def Setup(self, bottom):
        super(NNResizeLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.NNResize(input, **self._param)
