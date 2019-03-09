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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.nn import Module, Parameter
from dragon.vm.torch.nn.modules.utils import _pair


class _DepthwiseConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias):
        super(_DepthwiseConvNd, self).__init__()
        if in_channels != out_channels:
            raise ValueError('in/out channels must be same')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = Parameter(Tensor(out_channels, 1, *kernel_size))
        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'DepthwiseConv{}d'.format(len(self.kernel_size)),
            'arguments': {
                'num_output': self.weight.shape[0],
                'kernel_shape': self.weight.shape[2:],
                'strides': _pair(self.stride),
                'pads': _pair(self.padding),
                'dilations': _pair(1),
                'data_format': 'NCHW',
            }
        }

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class DepthwiseConv2d(_DepthwiseConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        super(DepthwiseConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
                stride, padding, _pair(0), bias)

    def forward(self, input):
        inputs = [input, self.weight] + ([self.bias] if self.bias else [])
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)