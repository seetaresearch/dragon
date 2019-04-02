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

"""The Implementation of the common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import ops as _ops
from ..layer import Layer as _Layer


class InnerProductLayer(_Layer):
    """The implementation of ``InnerProductLayer``.

    Parameters
    ----------
    num_output : int
         The output dim. Refer `InnerProductParameter.num_output`_.
    bias_term : boolean
         Whether to use bias. Refer `InnerProductParameter.bias_term`_.
    weight_filler : FillerParameter
         The filler of weight. Refer `InnerProductParameter.weight_filler`_.
    bias_filler : FillerParameter
         The filler of bias. Refer `InnerProductParameter.bias_filler`_.
    axis : int
        The start axis to calculate. Refer `InnerProductParameter.axis`_.
    transpose : boolean
        Whether to transpose the weights. Refer `InnerProductParameter.transpose`_.

    """
    def __init__(self, LayerParameter):
        super(InnerProductLayer, self).__init__(LayerParameter)
        param = LayerParameter.inner_product_param
        self.arguments = {
            'axis': param.axis,
            'num_output': param.num_output,
            'transW': not param.transpose,
        }
        # Add weights and biases
        self.AddBlob(filler=self.GetFiller(param, 'weight_filler'))
        if param.bias_term:
            self.AddBlob(filler=self.GetFiller(param, 'bias_filler'))

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return _ops.FullyConnected(inputs, **self.arguments)


class AccuracyLayer(_Layer):
    """The implementation of ``AccuracyLayer``.

    Parameters
    ----------
    top_k : int
        The top-k accuracy. Refer `AccuracyParameter.top_k`_.
    axis : int
        The axis of classes. Refer `AccuracyParameter.axis`_.
    ignore_label : int
        The label to ignore. Refer `AccuracyParameter.ignore_label`_.

    """
    def __init__(self, LayerParameter):
        super(AccuracyLayer, self).__init__(LayerParameter)
        param = LayerParameter.accuracy_param
        self.arguments = {
            'top_k': param.top_k,
            'ignore_labels': [param.ignore_label]
                if param.HasField('ignore_label') else [],
        }

    def LayerSetup(self, bottom):
        return _ops.Accuracy(bottom, **self.arguments)


class PythonLayer(_Layer):
    """The implementation of ``PythonLayer``.

    Parameters
    ----------
    module : str
        The module. Refer `PythonParameter.module`_.
    layer : str
        The class name of layer. Refer `PythonParameter.layer`_.
    param_str : str
        The str describing parameters. Refer `PythonParameter.param_str`_.

    """
    def __init__(self, LayerParameter):
        super(PythonLayer, self).__init__(LayerParameter)
        param = LayerParameter.python_param
        self.arguments = {
            'module': param.module,
            'op': param.layer,
            'param_str': param.param_str,
            'num_outputs': len(self._top),
        }

    def LayerSetup(self, bottom):
        return _ops.Run(bottom, **self.arguments)


class EltwiseLayer(_Layer):
    """The implementation of ``EltwiseLayer``.

    Parameters
    ----------
    operation : EltwiseParameter.EltwiseOp
        The operation. Refer `EltwiseParameter.operation`_.
    coeff : list of float
        The coefficients. Refer `EltwiseParameter.coeff`_.

    """
    def __init__(self, LayerParameter):
        super(EltwiseLayer, self).__init__(LayerParameter)
        param = LayerParameter.eltwise_param
        self.arguments = {
            'operation': {0: 'PROD', 1: 'SUM', 2: 'MAX'}[param.operation],
            'coefficients': [element for element in param.coeff]
                if len(param.coeff) > 0 else None,
        }

    def LayerSetup(self, bottom):
        return _ops.Eltwise(bottom, **self.arguments)


class AddLayer(_Layer):
    """The extended implementation of ``EltwiseLayer``."""

    def __init__(self, LayerParameter):
        super(AddLayer, self).__init__(LayerParameter)

    def LayerSetup(self, bottom):
        return _ops.Add(bottom, **self.arguments)


class ConcatLayer(_Layer):
    """The implementation of ``ConcatLayer``.

    Parameters
    ----------
    axis : int
        The axis to concatenate. Refer `ConcatParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ConcatLayer, self).__init__(LayerParameter)
        self.arguments = {'axis': LayerParameter.concat_param.axis}

    def LayerSetup(self, bottom):
        return _ops.Concat(bottom, **self.arguments)


class SliceLayer(_Layer):
    """The implementation of ``SliceLayer``.

    Parameters
    ----------
    axis : int
        The axis to concatenate. Refer ``SliceParameter.axis``.
    slice_point : sequence of int
        The optional slice points. Refer ``SliceParameter.slice_point``.

    """
    def __init__(self, LayerParameter):
        super(SliceLayer, self).__init__(LayerParameter)
        slice_param = LayerParameter.slice_param
        self.arguments = {
            'axis': slice_param.axis,
            'slice_point': [int(e) for e in slice_param.slice_point],
            'num_outputs': len(self._top),
        }

    def LayerSetup(self, bottom):
        return _ops.Slice(bottom, **self.arguments)


class CropLayer(_Layer):
    """The implementation of ``CropLayer``.

    Parameters
    ----------
    axis : int
        The start axis. Refer `CropParameter.axis`_.
    offset : sequence of int
        The offsets. Refer `CropParameter.offset`_.

    """
    def __init__(self, LayerParameter):
        super(CropLayer, self).__init__(LayerParameter)
        param = LayerParameter.crop_param
        self.arguments = {
            'start_axis': param.axis,
            'offsets': [int(element) for element in param.offset],
        }

    def LayerSetup(self, bottom):
        if not isinstance(bottom, (tuple, list)) or len(bottom) != 2:
            raise ValueError('Excepted two bottom blobs.')
        self.arguments['shape_like'] = bottom[1]
        self.arguments['starts'] = self.arguments['sizes'] = None
        return _ops.Crop(bottom[0], **self.arguments)


class ReshapeLayer(_Layer):
    """The implementation of ``ReshapeLayer``.

    Parameters
    ----------
    shape : sequence of int
        The output shape. Refer `ReshapeParameter.shape`_.

    """
    def __init__(self, LayerParameter):
        super(ReshapeLayer, self).__init__(LayerParameter)
        self.arguments = {'shape': [int(element) for element
            in LayerParameter.reshape_param.shape.dim]}

    def LayerSetup(self, bottom):
        return _ops.Reshape(bottom, **self.arguments)


class PermuteLayer(_Layer):
    """The implementation of ``PermuteLayer``.

    Parameters
    ----------
    order : sequence of int
        The permutation. Refer `PermuteParameter.order`_.

    """
    def __init__(self, LayerParameter):
        super(PermuteLayer, self).__init__(LayerParameter)
        self.arguments = {'perm': [int(element) for element
            in LayerParameter.permute_param.order]}

    def LayerSetup(self, bottom):
        return _ops.Transpose(bottom, **self.arguments)


class FlattenLayer(_Layer):
    """The implementation of ``FlattenLayer``.

    Parameters
    ----------
    axis : int
        The start axis. Refer `FlattenParameter.axis`_.
    end_axis : int
        The end axis. Refer `FlattenParameter.end_axis`_.

    """
    def __init__(self, LayerParameter):
        super(FlattenLayer, self).__init__(LayerParameter)
        param = LayerParameter.flatten_param
        axis = param.axis; end_axis = param.end_axis
        num_axes = end_axis - axis + 1 if end_axis != -1 else -1
        self.arguments = {'axis': axis, 'num_axes': num_axes}

    def LayerSetup(self, bottom):
        return _ops.Flatten(bottom, **self.arguments)


class GatherLayer(_Layer):
    """The extended implementation of ``GatherOp``.

    Parameters
    ----------
    axis : int
        The axis for gathering. Refer ``GatherParameter.axis``.

    """
    def __init__(self, LayerParameter):
        super(GatherLayer, self).__init__(LayerParameter)
        self.arguments = {'axis': LayerParameter.gather_param.axis}

    def LayerSetup(self, bottom):
        if not isinstance(bottom, (tuple, list)) or len(bottom) != 2:
            raise ValueError('Excepted two bottom blobs.')
        return _ops.Gather(bottom[0], indices=bottom[1], **self.arguments)


class SoftmaxLayer(_Layer):
    """The implementation of ``SoftmaxLayer``.

    Parameters
    ----------
    axis : int
        The axis to perform softmax. Refer `SoftmaxParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(SoftmaxLayer, self).__init__(LayerParameter)
        self.arguments = {'axis': LayerParameter.softmax_param.axis}

    def LayerSetup(self, bottom):
        return _ops.Softmax(bottom, **self.arguments)


class ArgMaxLayer(_Layer):
    """The implementation of ``ArgMaxLayer``.

    Parameters
    ----------
    top_k : int
        The top k results to keep. Refer `ArgMaxParameter.top_k`_.
    axis : int
        The axis to perform argmax. Refer `ArgMaxParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ArgMaxLayer, self).__init__(LayerParameter)
        param = LayerParameter.argmax_param
        self.arguments = {
            'top_k': param.top_k,
            'axis': param.axis,
            'keep_dims': True,
        }

    def LayerSetup(self, bottom):
        return _ops.ArgMax(bottom, **self.arguments)


class BatchNormLayer(_Layer):
    """The implementation of ``BatchNormLayer``.

    Parameters
    ----------
    use_global_stats : boolean
        Refer `BatchNormParameter.use_global_stats`_.
    moving_average_fraction : float
        Refer `BatchNormParameter.moving_average_fraction`_.
    eps : float
        Refer `BatchNormParameter.eps`_.

    """
    def __init__(self, LayerParameter):
        super(BatchNormLayer, self).__init__(LayerParameter)
        param = LayerParameter.batch_norm_param
        self.arguments = {
            'use_stats': int(param.use_global_stats)
                if param.HasField('use_global_stats') else -1,
            'momentum': param.moving_average_fraction,
            'eps': param.eps,
            'axis': 1,
        }
        self.AddBlob(value=0, enforce_no_grad=True) # mean
        self.AddBlob(value=0, enforce_no_grad=True) # var
        self.AddBlob(value=1, enforce_no_grad=True) # gamma
        self.AddBlob(value=0, enforce_no_grad=True) # beta

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return _ops.BatchNorm(inputs, **self.arguments)


class GroupNormLayer(_Layer):
    """The implementation of ``GroupNormLayer``.

    Parameters
    ----------
    group : int
        Refer ``GroupNormParameter.group``.
    eps : float
        Refer ``GroupNormParameter.eps``.

    """
    def __init__(self, LayerParameter):
        super(GroupNormLayer, self).__init__(LayerParameter)
        param = LayerParameter.group_norm_param
        self.arguments = {
            'axis': 1,
            'group': int(param.group),
            'eps': param.eps,
        }
        self.AddBlob(value=1, enforce_no_grad=True) # gamma
        self.AddBlob(value=0, enforce_no_grad=True) # beta

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return _ops.GroupNorm(inputs, **self.arguments)


class InstanceNormLayer(_Layer):
    """The implementation of ``InstanceNormLayer``.

    Introduced by `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    Parameters
    ----------
    eps : float
        Refer ``InstanceNormParameter.eps``.

    """
    def __init__(self, LayerParameter):
        super(InstanceNormLayer, self).__init__(LayerParameter)
        self.arguments = {'eps': LayerParameter.instance_norm_param.eps, 'axis': 1}

    def LayerSetup(self, bottom):
        return _ops.InstanceNorm(bottom, **self.arguments)


class ScaleLayer(_Layer):
    """The implementation of ``ScaleLayer``.

    Parameters
    ----------
    axis : int
        The start axis. Refer `ScaleParameter.axis`_.
    num_axes : int
        The number of axes. Refer `ScaleParameter.num_axes`_.
    filler : FillerParameter
        The filler of scale parameter. Refer `ScaleParameter.filler`_.
    bias_term : boolean
        Whether to use bias. Refer `ScaleParameter.bias_term`_.
    bias_filler : FillerParameter
        The filler of bias parameter. Refer `ScaleParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(ScaleLayer, self).__init__(LayerParameter)
        param = LayerParameter.scale_param
        self.arguments = {
            'axis': param.axis,
            'num_axes': param.num_axes,
        }
        # Add weights and biases
        self.AddBlob(filler=self.GetFiller(param, 'filler'), value=1)
        if param.bias_term:
            self.AddBlob(filler=self.GetFiller(param, 'bias_filler'))

    def LayerSetup(self, bottom):
        inputs = [bottom]+ [blob['data'] for blob in self._blobs]
        return _ops.Affine(inputs, **self.arguments)


class BNLayer(_Layer):
    """The implementation of ``BNLayer``.

    Parameters
    ----------
    use_global_stats : boolean
        Refer `BatchNormParameter.use_global_stats`_.
    moving_average_fraction : float
        Refer `BatchNormParameter.moving_average_fraction`_.
    eps : float
        Refer `BatchNormParameter.eps`_.
    filler : FillerParameter
        The filler of scale parameter. Refer `ScaleParameter.filler`_.
    bias_filler : FillerParameter
        The filler of bias parameter. Refer `ScaleParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(BNLayer, self).__init__(LayerParameter)
        bn_param = LayerParameter.batch_norm_param
        scale_param = LayerParameter.scale_param
        self.arguments = {
            'axis': 1,
            'momentum': bn_param.moving_average_fraction,
            'eps': bn_param.eps,
            'use_stats': int(bn_param.use_global_stats)
                if bn_param.HasField('use_global_stats') else -1,
        }
        self.AddBlob(value=0, enforce_no_grad=True) # mean
        self.AddBlob(value=0, enforce_no_grad=True) # var
        self.AddBlob(filler=self.GetFiller(scale_param, 'filler'), value=1) # gamma
        self.AddBlob(filler=self.GetFiller(scale_param, 'bias_filler')) # beta

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return _ops.BatchNorm(inputs, **self.arguments)


class GNLayer(_Layer):
    """The implementation of ``GNLayer``.

    Parameters
    ----------
    group : int
        Refer ``GroupNormParameter.group``.
    eps : float
        Refer ``GroupNormParameter.eps``.
    filler : FillerParameter
        The filler of scale parameter. Refer `ScaleParameter.filler`_.
    bias_filler : FillerParameter
        The filler of bias parameter. Refer `ScaleParameter.bias_filler`_.

    """
    def __init__(self, LayerParameter):
        super(GNLayer, self).__init__(LayerParameter)
        gn_param = LayerParameter.group_norm_param
        scale_param = LayerParameter.scale_param
        self.arguments = {
            'axis': 1,
            'group': int(gn_param.group),
            'eps': gn_param.eps,
        }
        self.AddBlob(filler=self.GetFiller(scale_param, 'filler'), value=1) # scale
        self.AddBlob(filler=self.GetFiller(scale_param, 'bias_filler')) # bias

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return _ops.GroupNorm(inputs, **self.arguments)


class NormalizeLayer(_Layer):
    """The implementation of ``NormalizeLayer``.

    Parameters
    ----------
    across_spatial : boolean
        Whether to stat spatially. Refer `NormalizeParameter.across_spatial`_.
    scale_filler : FillerParameter
        The filler of scale parameter. Refer `NormalizeParameter.scale_filler`_.
    channel_shared : boolean
        Whether to scale across channels. Refer `NormalizeParameter.channel_shared`_.
    eps : float
        The eps. Refer `NormalizeParameter.eps`_.

    """
    def __init__(self, LayerParameter):
        super(NormalizeLayer, self).__init__(LayerParameter)
        param = LayerParameter.normalize_param
        self.l2norm_arguments = {
            'axis': 1,
            'num_axes': -1 if param.across_spatial else 1,
            'eps': param.eps,
        }
        self.affine_arguments = {
            'axis': 1,
            'num_axes': 0 if param.channel_shared else 1,
        }
        self.AddBlob(filler=self.GetFiller(param, 'scale_filler'), value=1) # scale

    def LayerSetup(self, bottom):
        norm_out = [_ops.L2Norm(bottom, **self.l2norm_arguments)]
        return _ops.Affine(
            norm_out + [blob['data'] for blob in self._blobs],
                **self.affine_arguments)


class TileLayer(_Layer):
    """The extended implementation of ``TileLayer``.

    Parameters
    ----------
    multiples : caffe_pb2.BlobShape
        The multiples. Refer `TileParameter.multiples`_.

    """
    def __init__(self, LayerParameter):
        super(TileLayer, self).__init__(LayerParameter)
        self.arguments = {
            'multiples': [int(multiple) for multiple
                in LayerParameter.tile_param.multiples.dim],
        }

    def LayerSetup(self, bottom):
        return _ops.Tile(bottom, **self.arguments)


class ReductionLayer(_Layer):
    """The extended implementation of ``ReductionLayer``.

    Parameters
    ----------
    operation : caffe_pb2.ReductionOp
        The operation. Refer `ReductionParameter.operation`_.
    axis : int
        The axis to to reduce. Refer `ReductionParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ReductionLayer, self).__init__(LayerParameter)
        param = LayerParameter.reduction_param
        if param.axis < 0:
            if param.axis != -1:
                raise ValueError('The negative axis can only be -1(reduce all).')
        self.arguments = {
            'operation': {1: 'SUM', 4: 'MEAN'}[param.operation],
            'axis': param.axis,
        }

    def LayerSetup(self, bottom):
        return _ops.Reduce(bottom, **self.arguments)


class ExpandDimsLayer(_Layer):
    """The implementation of ``ExpandDimsLayer``.

    Parameters
    ----------
    axis : int
        This axis to expand at. Refer `ExpandDimsParameter.axis`_.

    """
    def __init__(self, LayerParameter):
        super(ExpandDimsLayer, self).__init__(LayerParameter)
        self.arguments = {'axis': LayerParameter.expand_dims_param.axis}

    def LayerSetup(self, bottom):
        return _ops.ExpandDims(bottom, **self.arguments)


class StopGradientLayer(_Layer):
    """The implementation of ``StopGradientLayer``."""

    def __init__(self, LayerParameter):
        super(StopGradientLayer, self).__init__(LayerParameter)

    def LayerSetup(self, bottom):
        return _ops.StopGradient(bottom, **self.arguments)


class ProposalLayer(_Layer):
    """The implementation of ``ProposalLayer``.

    Parameters
    ----------
    stride : sequence of int
        The stride of anchors. Refer ``ProposalParameter.stride``.
    scale : sequence of float
        The scales of anchors. Refer `ProposalParameter.scale`_.
    ratio : sequence of float
        The ratios of anchors. Refer `ProposalParameter.ratio`_.
    pre_nms_top_n : int
        The num of anchors before nms. Refer `ProposalParameter.pre_nms_topn`_.
    post_nms_top_n : int
        The num of anchors after nms. Refer `ProposalParameter.post_nms_topn`_.
    nms_thresh : float
        The threshold of nms. Refer `ProposalParameter.nms_thresh`_.
    min_size : int
        The min size of anchors. Refer `ProposalParameter.min_size`_.
    min_level : int
        Finest level of the FPN pyramid. Refer ``ProposalParameter.min_level``.
    max_level : int
        Coarsest level of the FPN pyramid. Refer ``ProposalParameter.max_level``.
    canonical_scale : int
        The baseline scale of mapping policy. Refer ``ProposalParameter.canonical_scale``.
    canonical_level : int
        Heuristic level of the canonical scale. Refer ``ProposalParameter.canonical_level``.

    """
    def __init__(self, LayerParameter):
        super(ProposalLayer, self).__init__(LayerParameter)
        param = LayerParameter.proposal_param
        self.arguments = {
            'strides': param.stride,
            'ratios': param.ratio,
            'scales': param.scale,
            'pre_nms_top_n': param.pre_nms_top_n,
            'post_nms_top_n': param.post_nms_top_n,
            'nms_thresh': param.nms_thresh,
            'min_size': param.min_size,
            'min_level': param.min_level,
            'max_level': param.max_level,
            'canonical_scale': param.canonical_scale,
            'canonical_level': param.canonical_level,
        }

    def LayerSetup(self, bottom):
        return _ops.Proposal(bottom, **self.arguments)


class CastLayer(_Layer):
    """The implementation of ``CastLayer``.

    Parameters
    ----------
    dtype : str
        The stride of anchors. Refer ``CastParameter.dtype``.

    """
    def __init__(self, LayerParameter):
        super(CastLayer, self).__init__(LayerParameter)
        param = LayerParameter.cast_param
        self.arguments = {'dtype': param.dtype.lower()}

    def LayerSetup(self, bottom):
        return _ops.Cast(bottom, **self.arguments)