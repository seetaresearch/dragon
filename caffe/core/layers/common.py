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
"""Common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.tensor import Tensor
from dragon.core.ops import activation_ops
from dragon.core.ops import array_ops
from dragon.core.ops import framework_ops
from dragon.core.ops import math_ops
from dragon.core.ops import metric_ops
from dragon.core.ops import normalization_ops
from dragon.vm.caffe.core.layer import Layer
from dragon.vm.caffe.core.proto import caffe_pb2


class Accuracy(Layer):
    """Compute the top-k accuracy.

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
        self.call_args = {
            'axis': param.axis,
            'top_k': param.top_k,
            'ignore_index': param.ignore_label
            if param.HasField('ignore_label') else None,
        }

    def __call__(self, bottom):
        return metric_ops.accuracy(bottom, **self.call_args)


class ArgMax(Layer):
    """Compute the index of maximum elements along the given axis.

    Examples:

    ```python
    layer {
      type: "ArgMax"
      bottom: "ip2"
      top: "cls"
      argmax_param {
        axis: 1
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(ArgMax, self).__init__(layer_param)
        param = layer_param.argmax_param
        if param.top_k != 1:
            raise ValueError('Top-k argmax is not supported.')
        self.call_args = {'axis': param.axis, 'keepdims': True}

    def __call__(self, bottom):
        return array_ops.argmax(bottom, **self.call_args)


class BatchNorm(Layer):
    """Apply the batch normalization.
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
        self.scale_layer = None
        self.call_args = {
            'use_stats': int(param.use_global_stats)
            if param.HasField('use_global_stats') else -1,
            'momentum': param.moving_average_fraction,
            'epsilon': param.eps,
            'axis': 1,
        }

    def from_proto(self, proto):
        if self._call_layer:
            return
        super(BatchNorm, self).from_proto(proto)
        div_factor = float(self.blobs[2]['data'])
        if div_factor > 1:
            for blob in self.blobs:
                impl = blob['data']._impl
                impl.FromNumpy(impl.ToNumpy() / div_factor, False)

    def build(self, bottom):
        weight_shape = [bottom.shape[1]]
        one_filler = caffe_pb2.FillerParameter(type='constant', value=1)
        zero_filler = caffe_pb2.FillerParameter(type='constant', value=0)
        self.add_blob(weight_shape, zero_filler, False)  # running mean
        self.add_blob(weight_shape, one_filler, False)  # running var
        self.add_blob((1,), one_filler, False)  # running num
        scale_layer = self.scale_layer
        if scale_layer and scale_layer.inplace:
            scale_layer.add_blob(weight_shape, scale_layer.filler)
            if scale_layer.bias_term:
                scale_layer.add_blob(weight_shape, scale_layer.bias_filler)
                self.bias = scale_layer._blobs[1]['data']
            else:
                self.add_blob(weight_shape, zero_filler, False)
                self.bias = self.blobs[3]['data']
            self.weight = scale_layer._blobs[0]['data']
            scale_layer.__call__ = lambda *args, **kwargs: None
        else:
            self.add_blob(weight_shape, one_filler, False)  # fixed gamma
            self.add_blob(weight_shape, zero_filler, False)  # fixed beta
            self.weight, self.bias = [blob['data'] for blob in self.blobs[3:5]]
        del self.blobs[3:]  # Avoid to save the fixed blobs

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        inputs = [bottom, self.weight, self.bias] + \
                 [blob['data'] for blob in self.blobs[:2]]
        return normalization_ops.batch_norm(inputs, **self.call_args)


class Concat(Layer):
    """Concatenate the inputs along the given axis.

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
        self.call_args = {'axis': layer_param.concat_param.axis}

    def __call__(self, bottom):
        return array_ops.concat(bottom, **self.call_args)


class Crop(Layer):
    """Select the elements according to the dimensions of second bottom.

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
        self.axis = param.axis
        self.call_args = {'starts': [0] * param.axis + [v for v in param.offset]}

    def __call__(self, bottom):
        if not isinstance(bottom, (tuple, list)) or len(bottom) != 2:
            raise ValueError('Excepted two bottom blobs.')
        sizes = array_ops.concat([array_ops.shape(bottom[0])[:self.axis],
                                  array_ops.shape(bottom[1])[self.axis:]])
        output = array_ops.slice(bottom[0], sizes=sizes, **self.call_args)
        try:
            output._shape = (bottom[0].shape[:self.axis] +
                             bottom[1].shape[self.axis:])
        except (TypeError, IndexError):
            pass
        return output


class Eltwise(Layer):
    """Compute the element-wise operation on the sequence of inputs.

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
            0: math_ops.mul,
            1: math_ops.add,
            2: math_ops.maximum,
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
    """Flatten the input along the given axes.

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
        self.call_args = {'axis': param.axis, 'end_axis': param.end_axis}

    def __call__(self, bottom):
        return array_ops.flatten(bottom, **self.call_args)


class InnerProduct(Layer):
    """Compute the dense matrix multiplication along the given axes.

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
        self.out_channels = param.num_output
        self.weight_filler = param.weight_filler
        self.bias_filler = param.bias_filler
        self.bias_term = param.bias_term
        self.call_args = {'transpose_b': not param.transpose}

    def build(self, bottom):
        weight_shape = [self.out_channels, bottom.shape[1]]
        self.add_blob(weight_shape, self.weight_filler)
        if self.bias_term:
            self.add_blob([self.out_channels], self.bias_filler)

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        inputs = [bottom] + [blob['data'] for blob in self.blobs]
        return math_ops.gemm(inputs, **self.call_args)


class Input(Layer):
    """Produce input blobs with shape and dtype.

    Examples:

    ```python
    layer {
      type: "Input"
      top: "data1"
      top: "data2"
      input_param {
        shape: { dim: 2 dim: 3 }
        shape: { dim: 2 dim: 3 dim: 3 }
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Input, self).__init__(layer_param)
        param = layer_param.input_param
        self.output_shapes = []
        for i in range(len(self.top)):
            self.output_shapes.append([e for e in param.shape[i].dim])

    def __call__(self, bottom):
        outputs = []
        for shape in self.output_shapes:
            outputs.append(Tensor(shape, symbolic=True))
        return outputs


class Normalize(Layer):
    """Apply the L2 normalization.
    `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.

    Examples:

    ```python
    layer {
      type: "Normalize"
      bottom: "conv4"
      top: "conv4/norm"
      normalize_param {
        across_spatial: false
        eps: 1e-12
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
        self.filler = caffe_pb2.FillerParameter(type='constant', value=1)
        self.filler = param.scale_filler if param.HasField('scale_filler') else self.filler
        self.norm_args = {'axis': 1,
                          'end_axis': -1 if param.across_spatial else 1,
                          'epsilon': param.eps}
        self.scale_args = {'axis': 1}

    def build(self, bottom):
        weight_shape = [bottom.shape[1]]
        self.add_blob(weight_shape, self.filler)

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        outputs = [normalization_ops.lp_normalize(bottom, **self.norm_args)]
        outputs += [blob['data'] for blob in self.blobs]
        return array_ops.channel_affine(outputs, **self.scale_args)


class Permute(Layer):
    """Permute the dimensions of input.

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
        self.call_args = {'perm': [e for e in param.order]}

    def __call__(self, bottom):
        return array_ops.transpose(bottom, **self.call_args)


class Python(Layer):
    """Wrap a python class into a layer.

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
        self.call_args = {
            'module_name': param.module,
            'class_name': param.layer,
            'kwargs_str': param.param_str,
            'num_outputs': len(self.top),
        }

    def __call__(self, bottom):
        return framework_ops.python_plugin(bottom, **self.call_args)


class Reduction(Layer):
    """Compute the reduction value along the given axis.

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
        self.call_args = {'axis': [param.axis]}
        self.reduction = {1: array_ops.sum, 4: array_ops.mean}[param.operation]

    def __call__(self, bottom):
        top = self.reduction(bottom, **self.call_args)
        if self.scale != 1:
            top *= self.scale
        return top


class Reshape(Layer):
    """Change the dimensions of input.

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
        self.call_args = {'shape': [dim for dim in param.shape.dim]}

    def __call__(self, bottom):
        return array_ops.reshape(bottom, **self.call_args)


class Scale(Layer):
    """Compute the affine transformation along the given axes.

    Examples:

    ```python
    layer {
      type: "Scale"
      bottom: "conv1/bn"
      top: "conv1/bn"
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
        self.axis = param.axis
        self.num_axes = param.num_axes
        end_axis = -1 if self.num_axes < 1 else self.axis + self.num_axes - 1
        self.call_args = {'axis': self.axis, 'end_axis': end_axis}
        self.filler = caffe_pb2.FillerParameter(type='constant', value=1)
        self.filler = param.filler if param.HasField('filler') else self.filler
        self.bias_filler = param.bias_filler
        self.bias_term = param.bias_term
        self.inplace = self.top[0] == self.bottom[0]

    def build(self, bottom):
        weight_shape = bottom.shape[self.axis:self.axis + self.num_axes]
        self.add_blob(weight_shape, self.filler)
        if self.bias_term:
            self.add_blob(weight_shape, self.bias_filler)

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        inputs = [bottom] + [blob['data'] for blob in self.blobs]
        return array_ops.channel_affine(inputs, **self.call_args)


class Slice(Layer):
    """Split the input into chunks along the given axis.

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
        self.call_args = {
            'axis': param.axis,
            'num_or_size_splits': len(self.top),
            'slice_point': [e for e in param.slice_point],
        }

    def __call__(self, bottom):
        return array_ops.split(bottom, **self.call_args)


class Softmax(Layer):
    """Apply the softmax function.

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
        self.call_args = {'axis': layer_param.softmax_param.axis}

    def __call__(self, bottom):
        return activation_ops.softmax(bottom, **self.call_args)


class StopGradient(Layer):
    """Return the identity of input with truncated gradient-flow.

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
        return framework_ops.stop_gradient(bottom)


class Tile(Layer):
    """Repeat the input according to the given axis.

    Examples:

    ```python
    layer {
        type: "Tile"
        bottom: "data"
        top: "output"
        tile_param {
          axis: 1
          tiles: 2
        }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Tile, self).__init__(layer_param)
        param = layer_param.tile_param
        repeats = [1] * (param.axis + 1)
        repeats[param.axis] = param.tiles
        self.call_args = {'repeats': repeats}

    def __call__(self, bottom):
        return array_ops.tile(bottom, **self.call_args)
