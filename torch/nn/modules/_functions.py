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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.torch.autograd import function


class _Activation(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(_Activation, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, input, inplace=False):
        out = input if inplace else self.alloc()
        return self.dispatch([input], [out])


class _ConvNd(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(_ConvNd, self).__init__(key, dev, **kwargs)
        self.num_output = kwargs.get('out_channels', 1)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.dilations = kwargs.get('dilations', 1)
        self.group = kwargs.get('group', None)
        self.output_padding = kwargs.get('output_padding', None)
        self.padding = None if self.output_padding is None else 'SAME'

    def attributes(self):
        return {
            'op_type': self.__class__.__name__,
            'arguments': {
                'kernel_shape': self.kernel_shape,
                'strides': self.strides,
                'pads': self.pads,
                'dilations': self.dilations,
                'padding': self.padding,
                'output_padding': self.output_padding,
                'group': self.group,
                'data_format': 'NCHW',
            }
        }

    def forward(self, input, weight, bias=None):
        inputs = [input, weight] + ([bias] if bias else [])
        return self.dispatch(inputs, [self.alloc()])


class _Loss(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(_Loss, self).__init__(key, dev, **kwargs)
        self.reduction = kwargs.get('reduction', 'mean').upper()

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class _PoolNd(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(_PoolNd, self).__init__(key, dev, **kwargs)
        self.kernel_shape = kwargs.get('kernel_shape', 1)
        self.strides = kwargs.get('strides', 1)
        self.pads = kwargs.get('pads', 0)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.mode = kwargs.get('mode', 'MAX')
        self.global_pooling = kwargs.get('global_pooling', False)

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
                'global_pooling': self.global_pooling,
            }
        }

    def forward(self, input):
        return self.dispatch([input], [self.alloc()])


class Affine(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Affine, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'Affine',
            'arguments': {
                'axis': 1,
                'num_axes': 1,
            }
        }

    def forward(self, input, weight, bias=None):
        inputs = [input, weight] + ([bias] if bias else [])
        return self.dispatch(inputs, [self.alloc()])


class BatchNorm(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(BatchNorm, self).__init__(key, dev, **kwargs)
        self.momentum = kwargs.get('momentum', 0.1)
        self.eps = kwargs.get('eps', 1e-5)
        self.training = kwargs.get('training', 'False')

    def attributes(self):
        return {
            'op_type': 'BatchNorm',
            'arguments': {
                'axis': 1,
                'momentum': 1. - self.momentum,
                'eps': self.eps,
                'use_stats': int(not self.training),
            }
        }

    def forward(self, input, running_mean, running_var, weight, bias):
        inputs = [input, weight, bias, running_mean, running_var]
        return self.dispatch(inputs, [self.alloc()])


class Conv2d(_ConvNd):
    def __init__(self, key, dev, **kwargs):
        super(Conv2d, self).__init__(key, dev, **kwargs)


class ConvTranspose2d(_ConvNd):
    def __init__(self, key, dev, **kwargs):
        super(ConvTranspose2d, self).__init__(key, dev, **kwargs)


class CTCLoss(_Loss):
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


class DepthwiseConv2d(_ConvNd):
    def __init__(self, key, dev, **kwargs):
        super(DepthwiseConv2d, self).__init__(key, dev, **kwargs)


class DropBlock2d(_Activation):
    def __init__(self, key, dev, **kwargs):
        super(DropBlock2d, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', 7)
        self.keep_prob = kwargs.get('keep_prob', 0.9)
        self.alpha = kwargs.get('alpha', 1.)
        self.decrement = kwargs.get('decrement', 0.)

    def attributes(self):
        return {
            'op_type': 'DropBlock2d',
            'arguments': {
                'block_size': self.block_size,
                'keep_prob': self.keep_prob,
                'alpha': self.alpha,
                'decrement': self.decrement,
                'data_format': 'NCHW',
            }
        }


class Dropout(_Activation):
    def __init__(self, key, dev, **kwargs):
        super(Dropout, self).__init__(key, dev, **kwargs)
        self.p = kwargs.get('p', 0.5)

    def attributes(self):
        return {'op_type': 'Dropout', 'arguments': {'prob': self.p}}


class DropPath(_Activation):
    def __init__(self, key, dev, **kwargs):
        super(DropPath, self).__init__(key, dev, **kwargs)
        self.p = kwargs.get('p', 0.2)
        self.increment = kwargs.get('increment', 0.)

    def attributes(self):
        return {
            'op_type': 'DropPath',
            'arguments': {
                'prob': self.p,
                'increment': self.increment,
            }
        }


class Elu(_Activation):
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


class Linear(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Linear, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'FullyConnected',
            'arguments': {
                'axis': -1,
                'transW': True,
            },
        }

    def forward(self, input, weight, bias=None, out=None):
        inputs = [input, weight] + ([bias] if bias else [])
        outputs = [out] if out else [self.alloc()]
        return self.dispatch(inputs, outputs)


class LocalResponseNorm(function.Function):
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
            }
        }

    def forward(self, input):
        return self.dispatch([input], [self.alloc()])


class GroupNorm(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(GroupNorm, self).__init__(key, dev, **kwargs)
        self.group = kwargs.get('group', 32)
        self.eps = kwargs.get('eps', 1e-5)

    def attributes(self):
        return {
            'op_type': 'GroupNorm',
            'arguments': {
                'group': self.group,
                'axis': 1,
                'eps': self.eps,
            }
        }

    def forward(self, input, weight, bias):
        return self.dispatch([input, weight, bias], [self.alloc()])


class L1Loss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(L1Loss, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'L1Loss',
            'arguments': {
                'scale': 1.,
                'reduction': self.reduction,
            }
        }


class L2Loss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(L2Loss, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'L2Loss',
            'arguments': {
                'scale': 2.,
                'reduction': self.reduction,
            }
        }


class LpNormalize(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(LpNormalize, self).__init__(key, dev, **kwargs)
        self.p = kwargs.get('p', 2)
        self.axis = kwargs.get('axis', 0)
        self.eps = kwargs.get('eps', 1e-12)

    def attributes(self):
        return {
            'op_type': 'LpNormalize',
            'arguments': {
                'p': self.p,
                'axis': self.axis,
                'eps': self.eps,
                'num_axes': 1,
                'reduction': 'SUM',
            }
        }

    def forward(self, input, out=None):
        out = out if out else self.alloc()
        return self.dispatch([input], [out])


class LSTMCell(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(LSTMCell, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'LSTMCell', 'arguments': {}}

    def forward(self, input, cx):
        outputs = [self.alloc() for _ in range(2)]
        return self.dispatch([input, cx], outputs)


class NLLLoss(_Loss):
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
            }
        }


class Pad(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(Pad, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)
        self.value = kwargs.get('value', 0.)
        self.mode = kwargs.get('mode', 'CONSTANT')

    def attributes(self):
        return {
            'op_type': 'Pad',
            'arguments': {
                'mode': self.mode,
                'value': self.value,
                'pads_descs': [
                    '${{HANDLE}}/pads[{}]'
                    .format(n) for n in range(self.ndim * 2)
                ],
            }
        }

    def feed(self, ws, handle, pads):
        for i, e in enumerate(pads):
            self.feed_arg(
                ws,
                '{}/pads[{}]'.format(handle, i),
                e, 'int64'
            )

    def forward(self, input, pads):
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, pads),
        )


class Pool2d(_PoolNd):
    def __init__(self, key, dev, **kwargs):
        super(Pool2d, self).__init__(key, dev, **kwargs)


class PRelu(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(PRelu, self).__init__(key, dev, **kwargs)

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


class Relu(_Activation):
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


class Relu6(_Activation):
    def __init__(self, key, dev, **kwargs):
        super(Relu6, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'Relu',
            'arguments': {
                'max_value': 6.,
            },
        }


class Resize(function.Function):
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
                'sizes_descs': [
                    '${{HANDLE}}/sizes[{}]'.format(n)
                    for n in range(self.num_sizes)
                ],
                'scales_descs': [
                    '${{HANDLE}}/scales[{}]'.format(n)
                    for n in range(self.num_scales)
                ],
            }
        }

    def feed(self, ws, handle, sizes, scales):
        for i in range(self.num_sizes):
            self.feed_arg(
                ws,
                '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64',
            )
        for i in range(self.num_scales):
            self.feed_arg(
                ws,
                '{}/scales[{}]'.format(handle, i),
                scales[i], 'float32',
            )

    def forward(self, input, sizes=None, scales=None):
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, sizes, scales)
        )


class RNNParamSet(function.Function):
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
        return self.dispatch(
            [param], [weights],
            no_grad=True,
            check_device=False,
        )


class SigmoidCrossEntropy(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SigmoidCrossEntropy, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'SigmoidCrossEntropy',
            'arguments': {
                'reduction': self.reduction,
            }
        }


class SigmoidFocalLoss(_Loss):
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


class SmoothL1Loss(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SmoothL1Loss, self).__init__(key, dev, **kwargs)
        self.beta = kwargs.get('beta', 1.)

    def attributes(self):
        return {
            'op_type': 'SmoothL1Loss',
            'arguments': {
                'beta': float(self.beta),
                'reduction': self.reduction,
            }
        }


class Softmax(_Activation):
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


class SoftmaxCrossEntropy(_Loss):
    def __init__(self, key, dev, **kwargs):
        super(SoftmaxCrossEntropy, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'SoftmaxCrossEntropy',
            'arguments': {
                'axis': 1,
                'reduction': self.reduction,
            }
        }


class SparseSoftmaxCrossEntropy(_Loss):
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
            }
        }


class SyncBatchNorm(BatchNorm):
    def __init__(self, key, dev, **kwargs):
        super(SyncBatchNorm, self).__init__(key, dev, **kwargs)
        self.process_group = kwargs.get('process_group', None)

    def attributes(self):
        attrs = BatchNorm.attributes(self)
        attrs['op_type'] = 'SyncBatchNorm'
        attrs['arguments'].update(self.process_group.arguments)
        return attrs
