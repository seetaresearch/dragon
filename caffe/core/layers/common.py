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

from dragon.core.autograph.tensor import TensorRef
from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.ops import activation_ops
from dragon.core.ops import array_ops
from dragon.core.ops import framework_ops
from dragon.core.ops import math_ops
from dragon.core.ops import metric_ops
from dragon.core.ops import normalization_ops
from dragon.vm.caffe.core.layer import Layer


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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Accuracy, self).__init__(layer_param)
        param = layer_param.accuracy_param
        self.arguments = {
            'axis': param.axis,
            'top_k': param.top_k,
            'ignore_index': param.ignore_label
            if param.HasField('ignore_label') else None,
        }

    def __call__(self, bottom):
        """
        Returns the metric for the given metric.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        return metric_ops.accuracy(bottom, **self.arguments)


class ArgMax(Layer):
    r"""Compute the index of maximum elements along the given axis.

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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(ArgMax, self).__init__(layer_param)
        param = layer_param.argmax_param
        if param.top_k != 1:
            raise ValueError('Top-k argmax is not supported.')
        self.arguments = {'axis': param.axis, 'keep_dims': True}

    def __call__(self, bottom):
        """
        Returns the argmax of the array.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(BatchNorm, self).__init__(layer_param)
        param = layer_param.batch_norm_param
        self.arguments = {
            'use_stats': int(param.use_global_stats)
            if param.HasField('use_global_stats') else -1,
            'momentum': param.moving_average_fraction,
            'epsilon': param.eps,
            'axis': 1,
        }
        self.add_blob(value=0, no_grad=True)  # running_mean
        self.add_blob(value=1, no_grad=True)  # running_var
        self.add_blob(value=1, no_grad=True)  # running_num
        self.add_blob(value=1, no_grad=True)  # fixed_gamma
        self.add_blob(value=0, no_grad=True)  # fixed_beta
        self._blobs[2]['data'].set_value([1.])
        self._weight, self._bias = [blob['data'] for blob in self._blobs[3:5]]
        del self._blobs[3:]  # Avoid to save the fixed blobs

    def fuse_with_scale_layer(self, scale_layer):
        """
        Fuse scale_layer to scale_layer.

        Args:
            self: (todo): write your description
            scale_layer: (todo): write your description
        """
        self._weight = scale_layer._blobs[0]['data']
        if len(scale_layer._blobs) == 2:
            self._bias = scale_layer._blobs[1]['data']
        scale_layer.__call__ = lambda *args, **kwargs: None

    def from_proto(self, proto):
        """
        Fetches the next tensor.

        Args:
            self: (todo): write your description
            proto: (todo): write your description
        """
        super(BatchNorm, self).from_proto(proto)
        current_ws = workspace.get_workspace()
        running_num = float(current_ws.fetch_tensor(self._blobs[2]['data']))
        if running_num != 1:
            running_mean = current_ws.fetch_tensor(self._blobs[0]['data'])
            running_var = current_ws.fetch_tensor(self._blobs[1]['data'])
            current_ws.feed_tensor(self._blobs[0]['data'], running_mean / running_num)
            current_ws.feed_tensor(self._blobs[1]['data'], running_var / running_num)
            current_ws.feed_tensor(self._blobs[2]['data'], [1], dtype='float32')

    def __call__(self, bottom):
        """
        Returns the model. _call.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        inputs = [bottom, self._weight, self._bias] + \
                 [blob['data'] for blob in self._blobs[:2]]
        return normalization_ops.batch_norm(inputs, **self.arguments)


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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Concat, self).__init__(layer_param)
        self.arguments = {'axis': layer_param.concat_param.axis}

    def __call__(self, bottom):
        """
        Concat. numpy. numpy. arrayops.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_paramoffs.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Crop, self).__init__(layer_param)
        param = layer_param.crop_param
        offsets = [e for e in param.offset]
        self.axis = param.axis
        self.arguments = {'starts': [[0] * param.axis] + offsets}

    def __call__(self, bottom):
        """
        Return the slice of the given axis.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Eltwise, self).__init__(layer_param)
        param = layer_param.eltwise_param
        self.eltwise_op = {
            0: math_ops.mul,
            1: math_ops.add,
            2: math_ops.maximum,
        }[param.operation]
        self.factors = [element for element in param.coeff]

    def __call__(self, bottom):
        """
        Return a factor of the factor.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_paramaxis.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Flatten, self).__init__(layer_param)
        param = layer_param.flatten_param
        axis, end_axis = param.axis, param.end_axis
        num_axes = end_axis - axis + 1 if end_axis != -1 else -1
        self.arguments = {'axis': axis, 'num_axes': num_axes}

    def __call__(self, bottom):
        """
        Call the array of - arrays.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        return array_ops.flatten(bottom, **self.arguments)


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
        """
        Initialize the layers.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(InnerProduct, self).__init__(layer_param)
        param = layer_param.inner_product_param
        self.arguments = {
            'axis': param.axis,
            'out_channels': param.num_output,
            'transpose_w': not param.transpose,
        }
        self.add_blob(filler=self.get_filler(param, 'weight_filler'))
        if param.bias_term:
            self.add_blob(filler=self.get_filler(param, 'bias_filler'))

    def __call__(self, bottom):
        """
        Call the module.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return math_ops.fully_connected(inputs, **self.arguments)


class Input(Layer):
    r"""Produce input blobs with shape and dtype.

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
        """
        Initialize layer inputs.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Input, self).__init__(layer_param)
        param = layer_param.input_param
        self.blob_shapes = []
        for i in range(len(self.top)):
            if i < len(param.shape):
                self.blob_shapes.append([e for e in param.shape[i].dim])
            else:
                self.blob_shapes.append(None)

    def __call__(self, bottom):
        """
        Implements object for the given route.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        name_scope = context.get_name_scope()
        current_ws = workspace.get_workspace()
        return [TensorRef(
            name=current_ws.unique_name(
                name_scope + 'output',
                suffix=':{}'.format(i),
                namespace='Tensor'),
            shape=self.blob_shapes[i],
            dtype='float32',
        ).constant() for i in range(len(self.blob_shapes))]


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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Normalize, self).__init__(layer_param)
        param = layer_param.normalize_param
        self.l2norm_arguments = {
            'axis': 1,
            'num_axes': -1 if param.across_spatial else 1,
            'epsilon': param.eps,
        }
        self.affine_arguments = {
            'axis': 1,
            'num_axes': 0 if param.channel_shared else 1,
        }
        self.add_blob(filler=self.get_filler(param, 'scale_filler'), value=1)

    def __call__(self, bottom):
        """
        Implement self ( l2 ) algorithm.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        norm_out = [normalization_ops.lp_normalize(bottom, **self.l2norm_arguments)]
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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Permute, self).__init__(layer_param)
        param = layer_param.permute_param
        self.arguments = {'perm': [e for e in param.order]}

    def __call__(self, bottom):
        """
        Compute the array.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize the layer

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Python, self).__init__(layer_param)
        param = layer_param.python_param
        self.arguments = {
            'module_name': param.module,
            'class_name': param.layer,
            'kwargs_str': param.param_str,
            'num_outputs': len(self.top),
        }

    def __call__(self, bottom):
        """
        Returns the arguments for the given plugin.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Reduction, self).__init__(layer_param)
        param = layer_param.reduction_param
        if param.axis < 0:
            if param.axis != -1:
                raise ValueError('The negative axis can only be -1.')
        self.scale = param.coeff
        self.arguments = {'axis': [param.axis]}
        self.reduction = {1: array_ops.sum, 4: array_ops.mean}[param.operation]

    def __call__(self, bottom):
        """
        Return the top - left and bottom parameters.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Reshape, self).__init__(layer_param)
        param = layer_param.reshape_param
        self.arguments = {'shape': [e for e in param.shape.dim]}

    def __call__(self, bottom):
        """
        Return the array of the arguments.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Scale, self).__init__(layer_param)
        param = layer_param.scale_param
        self.arguments = {'axis': param.axis, 'num_axes': param.num_axes}
        self.add_blob(filler=self.get_filler(param, 'filler'), value=1)
        if param.bias_term:
            self.add_blob(filler=self.get_filler(param, 'bias_filler'))

    def __call__(self, bottom):
        """
        Returns the module.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_paramuments.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Slice, self).__init__(layer_param)
        param = layer_param.slice_param
        self.arguments = {
            'axis': param.axis,
            'num_or_size_splits': len(self.top),
            'slice_point': [e for e in param.slice_point],
        }

    def __call__(self, bottom):
        """
        Return a callable for a callable.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        return array_ops.split(bottom, **self.arguments)


class Softmax(Layer):
    r"""Apply the softmax function.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

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
        """
        Initialize layer_paramuments.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Softmax, self).__init__(layer_param)
        self.arguments = {'axis': layer_param.softmax_param.axis}

    def __call__(self, bottom):
        """
        Call the l { layer with the given arguments.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
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
        """
        Initialize layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(StopGradient, self).__init__(layer_param)

    def __call__(self, bottom):
        """
        Call this cell.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        return framework_ops.stop_gradient(bottom, **self.arguments)


class Tile(Layer):
    r"""Repeat the input according to the given axis.

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
        """
        Initialize the layer layer_parameters.

        Args:
            self: (todo): write your description
            layer_param: (todo): write your description
        """
        super(Tile, self).__init__(layer_param)
        param = layer_param.tile_param
        repeats = [1] * (param.axis + 1)
        repeats[param.axis] = param.tiles
        self.arguments = {'repeats': repeats}

    def __call__(self, bottom):
        """
        Return a new tiles : class : class : array.

        Args:
            self: (todo): write your description
            bottom: (todo): write your description
        """
        return array_ops.tile(bottom, **self.arguments)
