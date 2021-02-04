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
"""NN functions library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.torch.core.autograd import function


class Activation(function.Function):
    """Activation function."""

    def __init__(self, key, dev, **kwargs):
        super(Activation, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, inplace=False):
        out = input if inplace else self.alloc()
        return self.dispatch([input], [out])


class Conv(function.Function):
    """Conv function."""

    def __init__(self, key, dev, **kwargs):
        super(Conv, self).__init__(key, dev, **kwargs)
        self.num_output = kwargs.get('out_channels', 1)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.dilations = kwargs.get('dilations', 1)
        self.group = kwargs.get('group', None)
        self.output_padding = kwargs.get('output_padding', None)

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'kernel_shape': self.kernel_shape,
                'strides': self.strides,
                'pads': self.pads,
                'dilations': self.dilations,
                'output_padding': self.output_padding,
                'group': self.group,
                'data_format': 'NCHW',
            }
        }

    def forward(self, input, weight, bias=None):
        inputs = [input, weight] + ([bias] if bias else [])
        return self.dispatch(inputs, [self.alloc()])


class Loss(function.Function):
    """Loss function."""

    def __init__(self, key, dev, **kwargs):
        super(Loss, self).__init__(key, dev, **kwargs)
        self.reduction = kwargs.get('reduction', 'mean').upper()

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Pool(function.Function):
    """Pool function."""

    def __init__(self, key, dev, **kwargs):
        super(Pool, self).__init__(key, dev, **kwargs)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.mode = kwargs.get('mode', 'MAX')

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'kernel_shape': self.kernel_shape,
                'strides': self.strides,
                'pads': self.pads,
                'ceil_mode': self.ceil_mode,
                'mode': self.mode,
                'data_format': 'NCHW',
            }
        }

    def forward(self, input):
        return self.dispatch([input], [self.alloc()])


class BatchNorm(function.Function):
    """BatchNorm function."""

    def __init__(self, key, dev, **kwargs):
        super(BatchNorm, self).__init__(key, dev, **kwargs)
        self.epsilon = kwargs.get('epsilon', 1e-5)
        self.training = kwargs.get('training', False)
        self.track_stats = kwargs.get('track_stats', True)

    def setup(self, ws, handle, momentum):
        self.feed_arg(ws, '{}/momentum'.format(handle), 1.0 - momentum, 'float32')

    def attributes(self):
        return {
            'op_type': 'BatchNorm',
            'arguments': {
                'axis': 1,
                'epsilon': self.epsilon,
                'use_stats': int(not self.training),
                'momentum_desc': '${HANDLE}/momentum',
            }
        }

    def forward(self, input, running_mean, running_var, weight, bias, momentum):
        inputs = [input, weight, bias, running_mean, running_var]
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, momentum),
        )


class ConvTranspose(Conv):
    """ConvTranspose function."""


class CTCLoss(Loss):
    """CTCLoss function."""

    def __init__(self, key, dev, **kwargs):
        super(CTCLoss, self).__init__(key, dev, **kwargs)
        self.padding_mask = kwargs.get('padding_mask', -1)

    def attributes(self):
        return {
            'op_type': 'CTCLoss',
            'arguments': {
                'padding_mask': self.padding_mask,
                'reduction': self.reduction,
            }
        }


class DepthwiseConv(Conv):
    """DepthwiseConv function."""


class Dropout(function.Function):
    """Dropout function."""

    def attributes(self):
        return {
            'op_type': 'Dropout',
            'arguments': {
                'ratio_desc': '${HANDLE}/ratio',
            },
        }

    def setup(self, ws, handle, ratio):
        self.feed_arg(ws, '{}/ratio'.format(handle), ratio, 'float32')

    def forward(self, input, ratio, inplace=False):
        out = input if inplace else self.alloc()
        return self.dispatch(
            [input], [out],
            callback=lambda ws, handle:
                self.setup(ws, handle, ratio),
        )


class DropBlock2d(Dropout):
    """DropBlock2d function."""

    def __init__(self, key, dev, **kwargs):
        super(DropBlock2d, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', 7)

    def attributes(self):
        return {
            'op_type': 'DropBlock2d',
            'arguments': {
                'data_format': 'NCHW',
                'block_size': self.block_size,
                'ratio_desc': '${HANDLE}/ratio',
            }
        }


class DropPath(Dropout):
    """DropPath function."""

    def attributes(self):
        return {
            'op_type': 'DropPath',
            'arguments': {
                'ratio_desc': '${HANDLE}/ratio',
            },
        }


class Elu(Activation):
    """ELU function."""

    def __init__(self, key, dev, **kwargs):
        super(Elu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)

    def attributes(self):
        return {
            'op_type': 'Elu',
            'arguments': {
                'alpha': float(self.alpha),
            },
        }


class GroupNorm(function.Function):
    """GroupNorm function."""

    def __init__(self, key, dev, **kwargs):
        super(GroupNorm, self).__init__(key, dev, **kwargs)
        self.group = kwargs.get('group', 32)
        self.epsilon = kwargs.get('epsilon', 1e-5)

    def attributes(self):
        return {
            'op_type': 'GroupNorm',
            'arguments': {
                'axis': 1,
                'group': self.group,
                'epsilon': self.epsilon,
            },
        }

    def forward(self, input, weight, bias):
        return self.dispatch([input, weight, bias], [self.alloc()])


class HardSigmoid(Activation):
    """HardSigmoid function."""

    def __init__(self, key, dev, **kwargs):
        super(HardSigmoid, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 0.5)

    def attributes(self):
        return {
            'op_type': 'HardSigmoid',
            'arguments': {
                'alpha': float(self.alpha),
                'beta': float(self.beta),
            },
        }


class HardSwish(Activation):
    """HardSwish function."""

    def __init__(self, key, dev, **kwargs):
        super(HardSwish, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 0.5)

    def attributes(self):
        return {
            'op_type': 'HardSwish',
            'arguments': {
                'alpha': float(self.alpha),
                'beta': float(self.beta),
            },
        }


class L1Loss(Loss):
    """L1Loss function."""

    def attributes(self):
        return {
            'op_type': 'L1Loss',
            'arguments': {
                'scale': 1.,
                'reduction': self.reduction,
            },
        }


class L2Loss(Loss):
    """L2Loss function."""

    def attributes(self):
        return {
            'op_type': 'L2Loss',
            'arguments': {
                'scale': 2.,
                'reduction': self.reduction,
            },
        }


class LocalResponseNorm(function.Function):
    """LocalResponseNorm function."""

    def __init__(self, key, dev, **kwargs):
        super(LocalResponseNorm, self).__init__(key, dev, **kwargs)
        self.size = kwargs.get('size', 5)
        self.alpha = kwargs.get('alpha', 0.0001)
        self.beta = kwargs.get('beta', 0.75)
        self.bias = kwargs.get('bias', 1.)

    def attributes(self):
        return {
            'op_type': 'LRN',
            'arguments': {
                'size': self.size,
                'alpha': self.alpha,
                'beta': self.beta,
                'bias': self.bias,
                'data_format': 'NCHW',
            },
        }

    def forward(self, input):
        return self.dispatch([input], [self.alloc()])


class LpNormalize(function.Function):
    """LpNormalize function."""

    def __init__(self, key, dev, **kwargs):
        super(LpNormalize, self).__init__(key, dev, **kwargs)
        self.p = kwargs.get('p', 2)
        self.axis = kwargs.get('axis', 0)
        self.epsilon = kwargs.get('epsilon', 1e-12)

    def attributes(self):
        return {
            'op_type': 'LpNormalize',
            'arguments': {
                'p': self.p,
                'axis': self.axis,
                'epsilon': self.epsilon,
                'num_axes': 1,
                'reduction': 'SUM',
            },
        }

    def forward(self, input, out=None):
        return self.dispatch([input], [self.alloc(out)])


class LSTMCell(function.Function):
    """LSTMCell function."""

    def forward(self, input, cx):
        outputs = [self.alloc() for _ in range(2)]
        return self.dispatch([input, cx], outputs)


class NLLLoss(Loss):
    """NLLLoss function."""

    def __init__(self, key, dev, **kwargs):
        super(NLLLoss, self).__init__(key, dev, **kwargs)
        self.ignore_index = kwargs.get('ignore_index', None)

    def attributes(self):
        return {
            'op_type': 'NLLLoss',
            'arguments': {
                'axis': 1,
                'reduction': self.reduction,
                'ignore_index': self.ignore_index,
            },
        }


class Pad(function.Function):
    """Pad function."""

    def __init__(self, key, dev, **kwargs):
        super(Pad, self).__init__(key, dev, **kwargs)
        self.value = kwargs.get('value', 0.)
        self.mode = kwargs.get('mode', 'CONSTANT')

    def attributes(self):
        return {
            'op_type': 'Pad',
            'arguments': {
                'mode': self.mode,
                'value': self.value,
                'pads_desc': '${HANDLE}/pads',
            },
        }

    def setup(self, ws, handle, pads):
        self.feed_arg(ws, '%s/pads' % handle, pads, 'int64')

    def forward(self, input, pads):
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, pads),
        )


class PRelu(function.Function):
    """PRelu function."""

    def attributes(self):
        return {
            'op_type': 'PRelu',
            'arguments': {
                'data_format': 'NCHW',
            },
        }

    def forward(self, input, weight):
        return self.dispatch([input, weight], [self.alloc()])


class Recurrent(function.Function):
    """Recurrent function."""

    def __init__(self, key, dev, **kwargs):
        super(Recurrent, self).__init__(key, dev, **kwargs)
        self.mode = kwargs.get('mode', 'rnn_tanh')
        self.num_layers = kwargs.get('num_layers', 1)
        self.hidden_size = kwargs.get('hidden_size', 0)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.dropout_ratio = kwargs.get('dropout_ratio', 0.)
        self.is_training = kwargs.get('is_training', False)
        self.num_outputs = 3 if 'lstm' in self.mode else 2

    def attributes(self):
        return {
            'op_type': 'Recurrent',
            'arguments': {
                'rnn_mode': self.mode,
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'bidirectional': self.bidirectional,
                'rnn_input_mode': 'linear',
                'dropout_ratio': self.dropout_ratio,
                'phase': 'TRAIN' if self.is_training else 'TEST'
            }
        }

    def forward(self, input, weights, hx=None):
        inputs = [input, weights]
        if hx is not None:
            inputs += nest.flatten(hx)
        outputs = [self.alloc() for _ in range(self.num_outputs)]
        outputs = self.dispatch(inputs, outputs)
        if self.num_outputs == 3:
            return outputs[0], outputs[1:]
        return outputs


class Relu(Activation):
    """Relu function."""

    def __init__(self, key, dev, **kwargs):
        super(Relu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.)

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {
                'alpha': float(self.alpha),
            },
        }


class Relu6(Activation):
    """Relu6 function."""

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {
                'max_value': 6.,
            },
        }


class Resize(function.Function):
    """Resize function."""

    def __init__(self, key, dev, **kwargs):
        super(Resize, self).__init__(key, dev, **kwargs)
        self.num_sizes = kwargs.get('num_sizes', 0)
        self.num_scales = kwargs.get('num_scales', 0)
        self.mode = kwargs.get('mode', 'NEAREST')
        self.mode = self.mode.replace('BILINEAR', 'LINEAR')
        self.mode = self.mode.replace('TRILINEAR', 'LINEAR')
        self.align_corners = kwargs.get('align_corners', False)

    def attributes(self):
        return {
            'op_type': 'Resize',
            'arguments': {
                'mode': self.mode,
                'align_corners': self.align_corners,
                'data_format': 'NCHW',
                'sizes_desc': '${HANDLE}/sizes'
                if self.num_sizes > 0 else None,
                'scales_desc': '${HANDLE}/scales'
                if self.num_scales > 0 else None,
            },
        }

    def setup(self, ws, handle, sizes, scales):
        if sizes is not None:
            self.feed_arg(ws, '%s/sizes' % handle, sizes, 'int64')
        if scales is not None:
            self.feed_arg(ws, '%s/scales' % handle, scales, 'float32')

    def forward(self, input, sizes=None, scales=None):
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, sizes, scales),
        )


class RNNParamSet(function.Function):
    """RNNParamSet function."""

    def __init__(self, key, dev, **kwargs):
        super(RNNParamSet, self).__init__(key, dev, **kwargs)
        self.param_type = kwargs.get('param_type', 'matrix')
        self.num_layers = kwargs.get('num_layers', 1)
        self.num_directions = kwargs.get('num_directions', 1)
        self.input_size = kwargs.get('input_size', 0)
        self.hidden_size = kwargs.get('hidden_size', 0)
        self.layer_id = kwargs.get('layer_id', 0)
        self.param_id = kwargs.get('param_id', 0)
        self.mode = kwargs.get('mode', 'rnn_tanh')

    def attributes(self):
        return {
            'op_type': 'RNNParamSet',
            'arguments': {
                'param_type': self.param_type,
                'num_layers': self.num_layers,
                'num_directions': self.num_directions,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'layer_id': self.layer_id,
                'param_id': self.param_id,
                'rnn_mode': self.mode,
            }
        }

    def forward(self, param, weights):
        return self.dispatch([param], [weights], no_grad=True)


class SigmoidCrossEntropy(Loss):
    """SigmoidCrossEntropy function."""

    def attributes(self):
        return {
            'op_type': 'SigmoidCrossEntropy',
            'arguments': {
                'reduction': self.reduction,
            },
        }


class SigmoidFocalLoss(Loss):
    """SigmoidFocalLoss function."""

    def __init__(self, key, dev, **kwargs):
        super(SigmoidFocalLoss, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.25)
        self.gamma = kwargs.get('gamma', 2.)
        self.negative_index = kwargs.get('negative_index', None)

    def attributes(self):
        return {
            'op_type': 'SigmoidFocalLoss',
            'arguments': {
                'axis': 1,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'negative_index': self.negative_index,
                'reduction': self.reduction,
            }
        }


class SmoothL1Loss(Loss):
    """SmoothL1Loss function."""

    def __init__(self, key, dev, **kwargs):
        super(SmoothL1Loss, self).__init__(key, dev, **kwargs)
        self.beta = kwargs.get('beta', 1.)

    def attributes(self):
        return {
            'op_type': 'SmoothL1Loss',
            'arguments': {
                'beta': float(self.beta),
                'reduction': self.reduction,
            },
        }


class Softmax(Activation):
    """Softmax function."""

    def __init__(self, key, dev, **kwargs):
        super(Softmax, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)

    def attributes(self):
        return {
            'op_type': 'Softmax',
            'arguments': {
                'axis': self.axis,
            },
        }


class SparseSoftmaxCrossEntropy(Loss):
    """SparseSoftmaxCrossEntropy function."""

    def __init__(self, key, dev, **kwargs):
        super(SparseSoftmaxCrossEntropy, self).__init__(key, dev, **kwargs)
        self.ignore_index = kwargs.get('ignore_index', None)

    def attributes(self):
        return {
            'op_type': 'SparseSoftmaxCrossEntropy',
            'arguments': {
                'axis': 1,
                'reduction': self.reduction,
                'ignore_index': self.ignore_index,
            },
        }


class SyncBatchNorm(BatchNorm):
    """SyncBatchNorm function."""

    def __init__(self, key, dev, **kwargs):
        super(SyncBatchNorm, self).__init__(key, dev, **kwargs)
        self.process_group = kwargs.get('process_group', None)

    def attributes(self):
        attrs = BatchNorm.attributes(self)
        if self.process_group is not None:
            attrs['op_type'] = 'SyncBatchNorm'
            attrs['arguments'].update(self.process_group.arguments)
        return attrs
