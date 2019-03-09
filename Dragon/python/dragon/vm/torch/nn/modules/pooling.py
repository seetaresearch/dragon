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

from dragon.vm.torch.nn import Module
from dragon.vm.torch.nn.modules.utils import _pair


class _PoolNd(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.register_op()

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, input):
        inputs = [input]; self.unify_devices(inputs)
        outputs = [self.register_output()]
        return self.run(inputs, outputs)


class MaxPool2d(_PoolNd):
    def register_op(self):
        self.op_meta = {
            'op_type': 'Pool2d',
            'arguments': {
                'kernel_shape': _pair(self.kernel_size),
                'strides': _pair(self.stride) if self.stride else _pair(self.kernel_size),
                'pads': _pair(self.padding),
                'mode': 'MAX',
                'data_format': 'NCHW',
                'ceil_mode': self.ceil_mode,
            }
        }


class AvgPool2d(_PoolNd):
    def register_op(self):
        self.op_meta = {
            'op_type': 'Pool2d',
            'arguments': {
                'kernel_shape': _pair(self.kernel_size),
                'strides': _pair(self.stride) if self.stride else _pair(self.kernel_size),
                'pads': _pair(self.padding),
                'mode': 'AVG',
                'data_format': 'NCHW',
                'ceil_mode': self.ceil_mode,
            }
        }