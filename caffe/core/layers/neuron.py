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
"""Neuron layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import activation_ops
from dragon.core.ops import math_ops
from dragon.vm.caffe.core.layer import Layer
from dragon.vm.caffe.core.proto import caffe_pb2


class Dropout(Layer):
    r"""Set the elements of the input to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * \text{Bernoulli}(p=1 - prob)

    Examples:

    ```python
    layer {
      type: "Dropout"
      bottom: "fc6"
      top: "fc6"
      dropout_param {
        dropout_ratio: 0.5
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Dropout, self).__init__(layer_param)
        param = layer_param.dropout_param
        if not param.scale_train:
            raise ValueError('Unscaled dropout is not supported.')
        self.call_args = {'ratio': param.dropout_ratio}

    def __call__(self, bottom):
        return activation_ops.dropout(bottom, **self.call_args)


class ELU(Layer):
    r"""Apply the exponential linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \text{ELU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    layer {
      type: "ELU"
      bottom: "conv2"
      top: "conv2"
      elu_param {
        alpha: 1.
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(ELU, self).__init__(layer_param)
        self.call_args = {'alpha': float(layer_param.elu_param.alpha)}

    def __call__(self, bottom):
        return activation_ops.elu(bottom, **self.call_args)


class Power(Layer):
    r"""Compute the power of input.

    .. math:: y = (scale * x + shift)^{power}

    Examples:

    ```python
    layer {
      type: "Power"
      bottom: "x"
      top: "y"
      power_param {
        scale: 1.
        shift: 0.
        power: 2.
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(Power, self).__init__(layer_param)
        param = layer_param.power_param
        self.scale = param.scale
        self.shift = param.shift
        self.power = param.power

    def __call__(self, bottom):
        if self.scale != 1:
            bottom = bottom * self.scale
        if self.shift != 0:
            bottom = bottom + self.shift
        return math_ops.pow([bottom, self.power])


class PReLU(Layer):
    r"""Apply the parametric rectified linear unit.
    `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    The **PReLU** function is defined as:

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
            x, & \text{ if } x \geq 0 \\
            weight * x, & \text{ otherwise }
        \end{cases}

        Examples:

    Examples:

    ```python
    layer {
      type: "PReLU"
      bottom: "conv2"
      top: "conv2/relu"
      prelu_param {
        channel_shared: false
        filler {
          type: "constant"
          value: 0.25
        }
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(PReLU, self).__init__(layer_param)
        param = layer_param.prelu_param
        self.filler = caffe_pb2.FillerParameter(type='constant', value=0.25)
        self.filler = param.filler if param.HasField('filler') else self.filler
        self.channel_shared = param.channel_shared

    def build(self, bottom):
        if self.channel_shared:
            weight_shape = [1]
        elif len(bottom.shape) > 1:
            weight_shape = [bottom.shape[1]]
        else:
            weight_shape = [bottom.shape[0]]
        self.add_blob(weight_shape, self.filler)

    def __call__(self, bottom):
        if len(self.blobs) == 0:
            self.build(bottom)
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return activation_ops.prelu(inputs)


class ReLU(Layer):
    r"""Apply the rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \text{ReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    layer {
      type: "ReLU"
      bottom: "conv2"
      top: "conv2/relu"
      relu_param {
        negative_slope: 0.
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(ReLU, self).__init__(layer_param)
        param = layer_param.relu_param
        self.negative_slope = param.negative_slope

    def __call__(self, bottom):
        if self.negative_slope > 0:
            return activation_ops.leaky_relu(bottom, self.negative_slope)
        return activation_ops.relu(bottom)


class Sigmoid(Layer):
    r"""Apply the sigmoid function.

    The **Sigmoid** function is defined as:

    .. math:: \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    layer {
      type: "Sigmoid"
      bottom: "rpn_cls_score"
      top: "rpn_cls_prob"
    }
    ```

    """

    def __init__(self, layer_param):
        super(Sigmoid, self).__init__(layer_param)

    def __call__(self, bottom):
        return activation_ops.sigmoid(bottom)


class TanH(Layer):
    r"""Apply the tanh function.

    The **Tanh** function is defined as:

    .. math:: \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}

    Examples:

    ```python
    layer {
      type: "TanH"
      bottom: "g/conv5"
      top: "g/image"
    }
    ```

    """

    def __init__(self, layer_param):
        super(TanH, self).__init__(layer_param)

    def __call__(self, bottom):
        return activation_ops.tanh(bottom)
