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

"""The implementation of the common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.tensor import Tensor
from dragon.core.ops import activation_ops
from dragon.core.ops import array_ops
from dragon.core.ops import framework_ops
from dragon.core.ops import math_ops
from dragon.core.ops import metric_ops
from dragon.core.ops import normalization_ops
from dragon.vm.caffe.layer import Layer


class Accuracy(Layer):
    r"""Compute the top-k accuracy.

    Examples:

    ```python
    layer {
        type: "Accuracy"
        bottom: "ip2"
        bottom: "label"
        top: "acc"
        accuracy_param {
            axis: 1
            top_k: 1
            ignore_label: -1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Accuracy, self).__init__(layer_param)
        param = layer_param.accuracy_param
        self.arguments = {
            'axis': param.axis,
            'top_k': param.top_k,
            'ignore_index': param.ignore_label
            if param.HasField('ignore_label') else None,
        }

    def __call__(self, bottom):
        return metric_ops.accuracy(bottom, **self.arguments)


class ArgMax(Layer):
    r"""Compute the indices of maximum elements along the given axis.

    Examples:

    ```python
    layer {
        type: "ArgMax"
        bottom: "ip2"
        top: "cls"
        argmax_param {
            top_k: 1
            axis: 1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(ArgMax, self).__init__(layer_param)
        param = layer_param.argmax_param
        self.arguments = {
            'top_k': param.top_k,
            'axis': param.axis,
            'keep_dims': True,
        }

    def __call__(self, bottom):
        return array_ops.argmax(bottom, **self.arguments)


class BatchNorm(Layer):
    r"""Apply the batch normalization.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    Examples:

    ```python
    layer {
        type: "BatchNorm"
        bottom: "conv1"
        top: "conv1/bn"
        batch_norm_param {
            use_global_stats: False
            moving_average_fraction: 0.9
            eps: 1e-5
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(BatchNorm, self).__init__(layer_param)
        param = layer_param.batch_norm_param
        self.arguments = {
            'use_stats': int(param.use_global_stats)
            if param.HasField('use_global_stats') else -1,
            'momentum': param.moving_average_fraction,
            'eps': param.eps,
            'axis': 1,
        }
        self.add_blob(value=1, no_grad=True)  # gamma
        self.add_blob(value=0, no_grad=True)  # beta
        self.add_blob(value=0, no_grad=True)  # running_mean
        self.add_blob(value=1, no_grad=True)  # running_var

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return normalization_ops.batch_norm(inputs, **self.arguments)


class Cast(Layer):
    r"""Cast the data type of input.

    Examples:

    ```python
    layer {
        type: "Cast"
        bottom: "ip2/fp16"
        top: "ip2/fp32"
        cast_param {
            dtype: "float32"
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Cast, self).__init__(layer_param)
        param = layer_param.cast_param
        self.arguments = {'dtype': param.dtype.lower()}

    def __call__(self, bottom):
        return array_ops.cast(bottom, **self.arguments)


class Concat(Layer):
    r"""Concatenate the inputs along the given axis.

    Examples:

    ```python
    layer {
        type: "Concat"
        bottom: "conv2"
        bottom: "conv1"
        top: "conv2/fuse"
        concat_param {
            axis: 1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Concat, self).__init__(layer_param)
        self.arguments = {'axis': layer_param.concat_param.axis}

    def __call__(self, bottom):
        return array_ops.concat(bottom, **self.arguments)


class Crop(Layer):
    r"""Select the elements according to the dimensions of second bottom.

    Examples:

    ```python
    layer {
        type: "Crop"
        bottom: "score"
        bottom: "score/ref"
        top: "score/crop"
        crop_param {
            axis: 2
            offset: 5
            offset: 10
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Crop, self).__init__(layer_param)
        param = layer_param.crop_param
        offsets = [e for e in param.offset]
        self.axis = param.axis
        self.arguments = {'starts': [[0] * param.axis] + offsets}

    def __call__(self, bottom):
        if not isinstance(bottom, (tuple, list)) or len(bottom) != 2:
            raise ValueError('Excepted two bottom blobs.')
        sizes = array_ops.concat([
            array_ops.shape(bottom[0])[:self.axis],
            array_ops.shape(bottom[1])[self.axis:],
        ])
        return array_ops.slice(bottom[0], sizes=sizes, **self.arguments)


class Eltwise(Layer):
    r"""Compute the element-wise operation on the sequence of inputs.

    Examples:

    ```python
    layer {
        type: "Eltwise"
        bottom: "conv2"
        bottom: "conv1"
        top: "conv2/fuse"
        eltwise_param {
            operation: SUM
            coeff: 1.
            coeff: 1.
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Eltwise, self).__init__(layer_param)
        param = layer_param.eltwise_param
        self.eltwise_op = {
            0: math_ops.mul,      # MUL
            1: math_ops.add,      # SUM
            2: math_ops.maximum,  # MAX
        }[param.operation]
        self.factors = [element for element in param.coeff]

    def __call__(self, bottom):
        for i in range(len(bottom)):
            if i < len(self.factors) and self.factors[i] != 1:
                bottom[i] *= self.factors[i]
        top = bottom[0]
        for i in range(1, len(bottom)):
            top = self.eltwise_op([top, bottom[i]])
        return top


class Flatten(Layer):
    r"""Flatten the input along the given axes.

    Examples:

    ```python
    layer {
        type: "Flatten"
        bottom: "conv5"
        top: "conv5/flatten"
        flatten_param {
            axis: 1
            end_axis: -1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Flatten, self).__init__(layer_param)
        param = layer_param.flatten_param
        axis, end_axis = param.axis, param.end_axis
        num_axes = end_axis - axis + 1 if end_axis != -1 else -1
        self.arguments = {'axis': axis, 'num_axes': num_axes}

    def __call__(self, bottom):
        return array_ops.flatten(bottom, **self.arguments)


class FusedBatchNorm(Layer):
    r"""Apply the fused batch normalization.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    Examples:

    ```python
    layer {
        type: "FusedBatchNorm"
        bottom: "conv1"
        top: "conv1/bn"
        batch_norm_param {
            use_global_stats: False
            moving_average_fraction: 0.9
            eps: 1e-5
        }
        scale_param {
            filler: {
                type: "constant"
                value: 1
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
        super(FusedBatchNorm, self).__init__(layer_param)
        bn_param = layer_param.batch_norm_param
        scale_param = layer_param.scale_param
        self.arguments = {
            'axis': 1,
            'momentum': bn_param.moving_average_fraction,
            'eps': bn_param.eps,
            'use_stats': int(bn_param.use_global_stats)
            if bn_param.HasField('use_global_stats') else -1,
        }
        self.add_blob(filler=self.get_filler(scale_param, 'filler'), value=1)  # gamma
        self.add_blob(filler=self.get_filler(scale_param, 'bias_filler'))  # beta
        self.add_blob(value=0, no_grad=True)  # running_mean
        self.add_blob(value=1, no_grad=True)  # running_var

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return normalization_ops.batch_norm(inputs, **self.arguments)


class FusedGroupNorm(Layer):
    r"""Apply the fused group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    Examples:

    ```python
    layer {
        type: "FusedGroupNorm"
        bottom: "conv1"
        top: "conv1/gn"
        group_norm_param {
            group: 32
            eps: 1e-5
        }
        scale_param {
            filler: {
                type: "constant"
                value: 1
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
        super(FusedGroupNorm, self).__init__(layer_param)
        gn_param = layer_param.group_norm_param
        scale_param = layer_param.scale_param
        self.arguments = {
            'axis': 1,
            'group': gn_param.group,
            'eps': gn_param.eps,
        }
        self.add_blob(filler=self.get_filler(scale_param, 'filler'), value=1)  # gamma
        self.add_blob(filler=self.get_filler(scale_param, 'bias_filler'))  # beta

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return normalization_ops.group_norm(inputs, **self.arguments)


class GroupNorm(Layer):
    r"""Apply the group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    Examples:

    ```python
    layer {
        type: "GroupNorm"
        bottom: "conv1"
        top: "conv1/gn"
        group_norm_param {
            group: 32
            eps: 1e-5
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(GroupNorm, self).__init__(layer_param)
        param = layer_param.group_norm_param
        self.arguments = {
            'axis': 1,
            'group': param.group,
            'eps': param.eps,
        }
        self.add_blob(value=1, no_grad=True)
        self.add_blob(value=0, no_grad=True)

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return normalization_ops.group_norm(inputs, **self.arguments)


class InnerProduct(Layer):
    r"""Compute the dense matrix multiplication along the given axes.

    Examples:

    ```python
    layer {
        type: "InnerProduct"
        bottom: "conv5"
        top: "ip1"
        inner_product_param {
            axis: 1
            num_output: 1024
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(InnerProduct, self).__init__(layer_param)
        param = layer_param.inner_product_param
        self.arguments = {
            'axis': param.axis,
            'num_output': param.num_output,
            'transW': not param.transpose,
        }
        # Add weights and biases
        self.add_blob(filler=self.get_filler(param, 'weight_filler'))
        if param.bias_term:
            self.add_blob(filler=self.get_filler(param, 'bias_filler'))

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return math_ops.fully_connected(inputs, **self.arguments)


class Input(Layer):
    r"""Produce input blobs with shape and dtype.

    Examples:

    ```python
    layer {
        type: "Input"
        top: "a"
        top: "b"
        input_param {
            shape: { dim: 2 dim: 3 }
            shape: { dim: 2 dim: 3 dim: 3 }
            dtype: "float32"
            dtype: "float64"
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Input, self).__init__(layer_param)
        param = layer_param.input_param
        self.shapes, self.dtypes = [], []
        for i in range(len(self.top)):
            if i < len(param.shape):
                self.shapes.append([e for e in param.shape[i].dim])
            else:
                self.shapes.append(None)
            if i < len(param.dtype):
                self.dtypes.append(param.dtype[i])
            else:
                self.dtypes.append('float32')

    def __call__(self, bottom):
        return [Tensor(shape=self.shapes[i], dtype=self.dtypes[i])
                for i in range(len(self.shapes))]


class Normalize(Layer):
    r"""Apply the fused L2 normalization.
    `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.

    Examples:

    ```python
    layer {
        type: "Normalize"
        bottom: "conv4"
        top: "conv4/norm"
        normalize_param {
            across_spatial: false
            channel_shared: false
            eps: 1e-5
            scale_filler: {
                type: "constant"
                value: 1
            }
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Normalize, self).__init__(layer_param)
        param = layer_param.normalize_param
        self.l2norm_arguments = {
            'axis': 1,
            'num_axes': -1 if param.across_spatial else 1,
            'eps': param.eps,
        }
        self.affine_arguments = {
            'axis': 1,
            'num_axes': 0 if param.channel_shared else 1,
        }
        self.add_blob(filler=self.get_filler(param, 'scale_filler'), value=1)

    def __call__(self, bottom):
        norm_out = [normalization_ops.l2_normalize(bottom, **self.l2norm_arguments)]
        norm_out += [blob['data'] for blob in self._blobs]
        return math_ops.affine(norm_out, **self.affine_arguments)


class Permute(Layer):
    r"""Permute the dimensions of input.

    Examples:

    ```python
    layer {
        type: "Permute"
        bottom: "cls_score"
        top: "cls_score/perm"
        permute_param {
            order: 0
            order: 2
            order: 3
            order: 1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Permute, self).__init__(layer_param)
        param = layer_param.permute_param
        self.arguments = {'perm': [e for e in param.order]}

    def __call__(self, bottom):
        return array_ops.transpose(bottom, **self.arguments)


class Python(Layer):
    r"""Wrap a python class into a layer.

    Examples:

    ```python
    layer {
        type: "Python"
        bottom: "cls_prob"
        bottom: "bbox_pred"
        bottom: "ims_info"
        top: "rois"
        python_param {
            module: 'rpn.proposal_layer'
            layer: 'ProposalLayer'
            param_str: "'feat_stride': 16"
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Python, self).__init__(layer_param)
        param = layer_param.python_param
        self.arguments = {
            'module_name': param.module,
            'class_name': param.layer,
            'kwargs_str': param.param_str,
            'num_outputs': len(self.top),
        }

    def __call__(self, bottom):
        return framework_ops.python_plugin(bottom, **self.arguments)


class Reduction(Layer):
    r"""Compute the reduction value along the given axis.

    Examples:

    ```python
    layer {
        type: "Reduction"
        bottom: "entropy"
        top: "loss"
        reduction_param {
            operation: SUM
            axis: 1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Reduction, self).__init__(layer_param)
        param = layer_param.reduction_param
        if param.axis < 0:
            if param.axis != -1:
                raise ValueError('The negative axis can only be -1.')
        self.scale = param.coeff
        self.arguments = {'axis': [param.axis]}
        self.reduction = {
            1: array_ops.sum,
            4: array_ops.mean,
        }[param.operation]

    def __call__(self, bottom):
        top = self.reduction(bottom, **self.arguments)
        if self.scale != 1:
            top *= self.scale
        return top


class Reshape(Layer):
    r"""Change the dimensions of input.

    Examples:

    ```python
    layer {
        type: "Reshape"
        bottom: "bbox_pred/perm"
        top: "bbox_pred/reshape"
        reshape_param {
            shape {
                dim: 0
                dim: -1
                dim: 4
            }
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Reshape, self).__init__(layer_param)
        param = layer_param.reshape_param
        self.arguments = {'shape': [e for e in param.shape.dim]}

    def __call__(self, bottom):
        return array_ops.reshape(bottom, **self.arguments)


class Scale(Layer):
    r"""Compute the affine transformation along the given axes.

    Examples:

    ```python
    layer {
        type: "Scale"
        bottom: "conv1/bn"
        top: "conv1/scale"
        scale_param {
            axis: 1
            num_axes: 1
            bias_term: true
            filler: {
                type: "constant"
                value: 1
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
        super(Scale, self).__init__(layer_param)
        param = layer_param.scale_param
        self.arguments = {'axis': param.axis, 'num_axes': param.num_axes}
        # Add weights and biases
        self.add_blob(filler=self.get_filler(param, 'filler'), value=1)
        if param.bias_term:
            self.add_blob(filler=self.get_filler(param, 'bias_filler'))

    def __call__(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return math_ops.affine(inputs, **self.arguments)


class Slice(Layer):
    r"""Split the input into chunks along the given axis.

    Examples:

    ```python
    layer {
        type: "Slice"
        bottom: "image"
        top: "image/b"
        top: "image/g"
        top: "image/r"
        slice_param {
            axis: 1
            slice_point: 1
            slice_point: 2
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Slice, self).__init__(layer_param)
        param = layer_param.slice_param
        self.arguments = {
            'axis': param.axis,
            'num_or_size_splits': len(self.top),
            'slice_point': [e for e in param.slice_point],
        }

    def __call__(self, bottom):
        return array_ops.split(bottom, **self.arguments)


class Softmax(Layer):
    r"""Apply the softmax function.

    Examples:

    ```python
    layer {
        type: "Softmax"
        bottom: "cls_score"
        top: "cls_prob"
        softmax_param {
            axis: 1
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Softmax, self).__init__(layer_param)
        self.arguments = {'axis': layer_param.softmax_param.axis}

    def __call__(self, bottom):
        return activation_ops.softmax(bottom, **self.arguments)


class StopGradient(Layer):
    r"""Return the identity of input with truncated gradient-flow.

    Examples:

    ```python
    layer {
        type: "StopGradient"
        bottom: "res2c"
        top: "res2c/frozen"
    }
    ```

    """

    def __init__(self, layer_param):
        super(StopGradient, self).__init__(layer_param)

    def __call__(self, bottom):
        return framework_ops.stop_gradient(bottom, **self.arguments)


class Tile(Layer):
    r"""Tile the input according to the given multiples.

    Examples:

    ```python
    layer {
        type: "Slice"
        bottom: "conv2"
        top: "conv2/dup"
        tile_param {
            multiples: {
                dim: 1
                dim: 2
                dim: 1
                dim: 1
            }
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Tile, self).__init__(layer_param)
        param = layer_param.tile_param
        self.arguments = {'multiples': [e for e in param.multiples.dim]}

    def __call__(self, bottom):
        return array_ops.tile(bottom, **self.arguments)