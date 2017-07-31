# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
from layer import Layer
import dragon.ops as ops

class InnerProductLayer(Layer):
    def __init__(self, LayerParameter):
        super(InnerProductLayer, self).__init__(LayerParameter)
        param = LayerParameter.inner_product_param
        self._param = {'axis': param.axis,
                       'num_output': param.num_output}
        weight = Tensor(LayerParameter.name + '@param0')
        weight_diff = Tensor(LayerParameter.name + '@param0_grad')
        self.Fill(weight, param, 'weight_filler')
        self._blobs.append({'data': weight, 'diff': weight_diff})

        if param.bias_term:
            bias = Tensor(LayerParameter.name + '@param1')
            bias_diff = Tensor(LayerParameter.name + '@param1_grad')
            self.Fill(bias, param, 'bias_filler')
            self._blobs.append({'data': bias, 'diff': bias_diff})

    def Setup(self, bottom):
        super(InnerProductLayer, self).Setup(bottom)
        return ops.InnerProduct(bottom + [blob['data'] for blob in self._blobs], **self._param)


class AccuracyLayer(Layer):
    def __init__(self, LayerParameter):
        super(AccuracyLayer, self).__init__(LayerParameter)
        param = LayerParameter.accuracy_param
        self._param = {'top_k': param.top_k,
                       'ignore_labels': [param.ignore_label]
                            if param.HasField('ignore_label') else []}

    def Setup(self, bottom):
        super(AccuracyLayer, self).Setup(bottom)
        return ops.Accuracy(bottom, **self._param)


class PythonLayer(Layer):
    def __init__(self, LayerParameter):
        super(PythonLayer, self).__init__(LayerParameter)
        param = LayerParameter.python_param
        self._param = {'module': param.module,
                       'op': param.layer,
                       'param_str': param.param_str}

    def Setup(self, bottom):
        super(PythonLayer, self).Setup(bottom)
        return ops.Run(bottom, nout=len(self._top), **self._param)


class EltwiseLayer(Layer):
    def __init__(self, LayerParameter):
        super(EltwiseLayer, self).__init__(LayerParameter)
        param = LayerParameter.eltwise_param
        self._param = {'operation': {0: 'PROD', 1: 'SUM', 2: 'MAX'}[param.operation],
                       'coeffs': [element for element in param.coeff]
                            if len(param.coeff) > 0 else None}

    def Setup(self, bottom):
        super(EltwiseLayer, self).Setup(bottom)
        return ops.Eltwise(bottom, **self._param)


class AddLayer(Layer):
    def __init__(self, LayerParameter):
        super(AddLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(AddLayer, self).Setup(bottom)
        return ops.Add(bottom, **self._param)


class ConcatLayer(Layer):
    def __init__(self, LayerParameter):
        super(ConcatLayer, self).__init__(LayerParameter)
        param = LayerParameter.concat_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(ConcatLayer, self).Setup(bottom)
        return ops.Concat(bottom, **self._param)


class DenseConcatLayer(Layer):
    def __init__(self, LayerParameter):
        super(DenseConcatLayer, self).__init__(LayerParameter)
        param = LayerParameter.concat_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(DenseConcatLayer, self).Setup(bottom)
        return ops.DenseConcat(bottom, **self._param)


class CropLayer(Layer):
    def __init__(self, LayerParameter):
        super(CropLayer, self).__init__(LayerParameter)
        param = LayerParameter.crop_param
        self._param = {'axis': param.axis,
                       'offsets': [int(element) for element in param.offset]}

    def Setup(self, bottom):
        super(CropLayer, self).Setup(bottom)
        self._param['shape_like'] = bottom[1]
        return ops.Crop(bottom[0], **self._param)


class ReshapeLayer(Layer):
    def __init__(self, LayerParameter):
        super(ReshapeLayer, self).__init__(LayerParameter)
        param = LayerParameter.reshape_param
        shape = param.shape
        self._param = {'shape': [int(element) for element in shape.dim]}

    def Setup(self, bottom):
        super(ReshapeLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Reshape(input, **self._param)


class PermuteLayer(Layer):

    """ Introduced in SSD by Wei Liu """

    def __init__(self, LayerParameter):
        super(PermuteLayer, self).__init__(LayerParameter)
        param = LayerParameter.permute_param
        self._param = {'perm': [int(element) for element in param.order]}

    def Setup(self, bottom):
        super(PermuteLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Transpose(input, **self._param)


class FlattenLayer(Layer):
    def __init__(self, LayerParameter):
        super(FlattenLayer, self).__init__(LayerParameter)
        param = LayerParameter.flatten_param
        axis = param.axis; end_axis = param.end_axis
        num_axes =  end_axis - axis + 1 if end_axis != -1 else -1
        self._param = {'axis': axis, 'num_axes': num_axes}

    def Setup(self, bottom):
        super(FlattenLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Flatten(input, **self._param)


class SoftmaxLayer(Layer):
    def __init__(self, LayerParameter):
        super(SoftmaxLayer, self).__init__(LayerParameter)
        param = LayerParameter.softmax_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(SoftmaxLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Softmax(input, **self._param)


class BatchNormLayer(Layer):
    def __init__(self, LayerParameter):
        super(BatchNormLayer, self).__init__(LayerParameter)
        param = LayerParameter.batch_norm_param
        self._param = {'use_stats': int(param.use_global_stats)
                            if param.HasField('use_global_stats') else -1,
                       'momentum': param.moving_average_fraction,
                       'eps': param.eps}
        # mean, var, factor are set to 0 in order to do statistics
        mean = Tensor(LayerParameter.name + '@param0').Constant()
        var  = Tensor(LayerParameter.name + '@param1').Constant()
        factor = Tensor(LayerParameter.name + '@param2').Constant()
        # in dragon, set diff as None will ignore computing grad automatically
        # but in bvlc-caffe1, you must set lr_mult = 0 manually
        self._blobs.append({'data': mean, 'diff': None})
        self._blobs.append({'data': var, 'diff': None})
        self._blobs.append({'data': factor, 'diff': None})

    def Setup(self, bottom):
        super(BatchNormLayer, self).Setup(bottom)
        return ops.BatchNorm(bottom + [blob['data'] for blob in self._blobs], **self._param)


class BatchRenormLayer(Layer):
    def __init__(self, LayerParameter):
        super(BatchRenormLayer, self).__init__(LayerParameter)
        param = LayerParameter.batch_renorm_param
        self._param = {'use_stats': int(param.use_global_stats)
                            if param.HasField('use_global_stats') else -1,
                       'momentum': param.moving_average_fraction,
                       'eps': param.eps,
                       'r_max': float(param.r_max),
                       'd_max': float(param.d_max),
                       't_delta': float(param.t_delta)}
        mean = Tensor(LayerParameter.name + '@param0').Constant()
        var  = Tensor(LayerParameter.name + '@param1').Constant()
        factor = Tensor(LayerParameter.name + '@param2').Constant()
        self._blobs.append({'data': mean, 'diff': None})
        self._blobs.append({'data': var, 'diff': None})
        self._blobs.append({'data': factor, 'diff': None})

    def Setup(self, bottom):
        super(BatchRenormLayer, self).Setup(bottom)
        return ops.BatchRenorm(bottom + [blob['data'] for blob in self._blobs], **self._param)


class InstanceNormLayer(Layer):
    def __init__(self, LayerParameter):
        super(InstanceNormLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(InstanceNormLayer, self).Setup(bottom)
        return ops.InstanceNorm(bottom[0], **self._param)


class ScaleLayer(Layer):
    def __init__(self, LayerParameter):
        super(ScaleLayer, self).__init__(LayerParameter)
        param = LayerParameter.scale_param
        self._param = {'axis': param.axis,
                       'num_axes': param.num_axes}
        scale = Tensor(LayerParameter.name + '@param0')
        scale_diff = Tensor(LayerParameter.name + '@param0_grad')
        if param.HasField('filler'):
            self.Fill(scale, param, 'filler')
        else: scale.Constant(value=1.0)
        self._blobs.append({'data': scale, 'diff': scale_diff})
        if param.bias_term:
            bias = Tensor(LayerParameter.name + '@param1')
            bias_diff = Tensor(LayerParameter.name + '@param1_grad')
            # auto fill 0 if not specficed bias_filler
            self.Fill(bias, param, 'bias_filler')
            self._blobs.append({'data': bias, 'diff': bias_diff})

    def Setup(self, bottom):
        super(ScaleLayer, self).Setup(bottom)
        return ops.Scale(bottom + [blob['data'] for blob in self._blobs], **self._param)


class BNLayer(Layer):

    """ Introduced in some forked caffe or CUDNN """

    def __init__(self, LayerParameter):
        super(BNLayer, self).__init__(LayerParameter)
        param = LayerParameter.batch_norm_param
        self._param = {'use_stats': int(param.use_global_stats)
                                        if param.HasField('use_global_stats') else -1,
                       'momentum': param.moving_average_fraction,
                       'eps': param.eps}
        mean = Tensor(LayerParameter.name + '@param0').Constant()
        var = Tensor(LayerParameter.name + '@param1').Constant()
        scale = Tensor(LayerParameter.name + '@param2').Uniform(low=0.0, high=1.0)
        bias = Tensor(LayerParameter.name + '@param3').Constant(value=0.0)
        self.norm_blobs = [{'data': mean, 'diff': None},
                           {'data': var, 'diff': None}]
        self.scale_blobs = [{'data': scale, 'diff': Tensor(scale.name + '_grad')},
                            {'data': bias, 'diff': Tensor(bias.name + '_grad')}]
        self._blobs.extend(self.norm_blobs)
        self._blobs.extend(self.scale_blobs)

    def Setup(self, bottom):
        super(BNLayer, self).Setup(bottom)
        return ops.BN(bottom + [blob['data'] for blob in self._blobs], **self._param)


class NormalizeLayer(Layer):

    """ Introduced in SSD by Wei Liu """

    def __init__(self, LayerParameter):
        super(NormalizeLayer, self).__init__(LayerParameter)
        param = LayerParameter.normalize_param
        self._l2norm_param = {'axis': 1,
                              'num_axes': -1 if param.across_spatial else 1,
                              'eps': param.eps}
        self._scale_param = {'axis': 1,
                             'num_axes': 0 if param.channel_shared else 1}
        scale = Tensor(LayerParameter.name + '@param0')
        if param.HasField('scale_filler'):
            self.Fill(scale, param, 'scale_filler')
        else: scale.Contant(value=1.0)
        self.scale_blobs = [{'data': scale, 'diff': Tensor(scale.name + '_grad')}]
        self._blobs.extend(self.scale_blobs)

    def Setup(self, bottom):
        super(NormalizeLayer, self).Setup(bottom)
        norm_out = [ops.L2Norm(bottom[0], **self._l2norm_param)]
        scale_out = ops.Scale(norm_out + [blob['data'] for blob in self.scale_blobs],
                              **self._scale_param)
        return scale_out


class TileLayer(Layer):
    def __init__(self, LayerParameter):
        super(TileLayer, self).__init__(LayerParameter)
        param = LayerParameter.tile_param
        multiples = param.multiples
        self._param = {'multiples': [int(multiple) for multiple in multiples.dim]}

    def Setup(self, bottom):
        super(TileLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Tile(input, **self._param)


class ExpandDimsLayer(Layer):
    def __init__(self, LayerParameter):
        super(ExpandDimsLayer, self).__init__(LayerParameter)
        param = LayerParameter.expand_dims_param
        self._param = {'axis': param.axis}

    def Setup(self, bottom):
        super(ExpandDimsLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.ExpandDims(input, **self._param)


class ProposalLayer(Layer):
    def __init__(self, LayerParameter):
        super(ProposalLayer, self).__init__(LayerParameter)
        param = LayerParameter.proposal_param
        self._param = {'base_size': param.base_size,
                       'min_size': param.min_size,
                       'feat_stride': param.feat_stride,
                       'pre_nms_topn': param.pre_nms_topn,
                       'post_nms_topn': param.post_nms_topn,
                       'nms_thresh': param.nms_thresh,
                       'ratio': param.ratio,
                       'scale': param.scale}

    def Setup(self, bottom):
        super(ProposalLayer, self).Setup(bottom)
        return ops.Proposal(bottom, **self._param)
