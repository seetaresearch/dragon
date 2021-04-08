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
"""Vision layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import normalization_ops
from dragon.core.ops import vision_ops
from dragon.vm.caffe.core.layer import Layer


class Convolution(Layer):
    r"""Apply the n-dimension convolution.

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
        self.kernel_shape = param.kernel_size or [1]
        self.strides = param.stride or [1]
        self.pads = param.pad or [0]
        self.dilations = param.dilation or [1]
        self.out_channels = param.num_output
        self.weight_filler = param.weight_filler
        self.bias_filler = param.bias_filler
        self.bias_term = param.bias_term
        self.call_args = {'group': param.group}

    def build(self, bottom):
        num_axes = len(bottom.shape) - 2
        if num_axes < 1:
            raise ValueError(
                'Bottom 0 of layer "{}" is {}d, excepted 3d/4d/5d.'
                .format(self.name, len(bottom.shape)))
        in_channels = bottom.shape[1]
        for k in ('kernel_shape', 'strides', 'pads', 'dilations'):
            self.call_args[k] = [int(d) for d in getattr(self, k)]
            if len(self.call_args[k]) < num_axes:
                reps = num_axes - len(self.call_args[k])
                self.call_args[k] += [self.call_args[k][-1]] * reps
        weight_shape = [self.out_channels, in_channels] + self.call_args['kernel_shape']
        self.add_blob(weight_shape, self.weight_filler)
        if self.bias_term:
            self.add_blob([self.out_channels], self.bias_filler)

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        conv_op = 'conv{}d'.format(len(self.call_args['kernel_shape']))
        return getattr(vision_ops, conv_op)(inputs, **self.call_args)


class Deconvolution(Convolution):
    r"""Apply the n-dimension deconvolution.

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

    def build(self, bottom):
        num_axes = len(bottom.shape) - 2
        if num_axes < 1:
            raise ValueError(
                'Bottom 0 of layer "{}" is {}d, excepted 3d/4d/5d.'
                .format(self.name, len(bottom.shape)))
        in_channels = bottom.shape[1]
        for k in ('kernel_shape', 'strides', 'pads', 'dilations'):
            self.call_args[k] = [int(d) for d in getattr(self, k)]
            if len(self.call_args[k]) < num_axes:
                reps = num_axes - len(self.call_args[k])
                self.call_args[k] += [self.call_args[k][-1]] * reps
        weight_shape = [in_channels, self.out_channels] + self.call_args['kernel_shape']
        self.add_blob(weight_shape, self.weight_filler)
        if self.bias_term:
            self.add_blob([self.out_channels], self.bias_filler)

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        conv_op = 'conv{}d_transpose'.format(len(self.call_args['kernel_shape']))
        return getattr(vision_ops, conv_op)(inputs, **self.call_args)


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
            raise NotImplementedError('<WITHIN_CHANNEL> mode is not implemented.')
        self.op_args = {'size': param.local_size,
                        'alpha': param.alpha,
                        'beta': param.beta,
                        'bias': param.k}

    def __call__(self, bottom):
        return normalization_ops.local_response_norm(bottom, **self.op_args)


class Pooling(Layer):
    r"""Apply the n-dimension pooling.

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
        self.kernel_shape = [param.kernel_size or 1]
        self.strides = [param.stride or 1]
        self.pads = [param.pad or 0]
        self.call_args = {
            'ceil_mode': True,
            'mode': {0: 'MAX', 1: 'AVG'}[param.pool],
            'global_pool': param.global_pooling,
        }

    def __call__(self, bottom):
        num_axes = len(bottom.shape) - 2
        if num_axes < 1:
            raise ValueError(
                'Bottom 0 of layer "{}" is {}d, excepted 3d/4d/5d.'
                .format(self.name, len(bottom.shape)))
        call_args = self.call_args.copy()
        for k in ('kernel_shape', 'strides', 'pads'):
            call_args[k] = [int(d) for d in getattr(self, k)]
            if len(call_args[k]) < num_axes:
                reps = num_axes - len(call_args[k])
                call_args[k] += [call_args[k][-1]] * reps
        pool_op = 'pool{}d'.format(num_axes)
        return getattr(vision_ops, pool_op)(bottom, **call_args)
