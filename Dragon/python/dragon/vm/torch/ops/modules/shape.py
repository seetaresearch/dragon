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

from dragon.vm.torch.ops.modules.base import BaseModule


class Fill(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Fill, self).__init__(key, ctx, **kwargs)
        self.len_shape = kwargs.get('len_shape', 0)
        self.value = kwargs.get('value', 0.0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.shape = [self.register_argument('shape[{}]'.format(i))
                for i in range(self.len_shape)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Fill',
            'n_inputs': 0, 'n_outputs': 1,
            'arguments': {
                'value': float(self.value),
                'dims_desc': [d for d in self.shape] if len(self.shape) > 0 else None,
            }
        }

    def forward(self, x, shape):
        outputs = [x]; self.unify_devices(outputs)
        if shape is not None:
            for ix, d in enumerate(shape):
                self.set_argument_i(self.shape[ix], d)
        return self.run([], outputs)


class Reshape(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Reshape, self).__init__(key, ctx, **kwargs)
        self.len_shape = kwargs.get('len_shape', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.shape = [self.register_argument('shape[{}]'.format(i))
                for i in range(self.len_shape)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Reshape',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'shape_desc': [d for d in self.shape]
                    if len(self.shape) > 0 else None,
            }
        }

    def forward(self, x, shape):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        if shape is not None:
            for ix, d in enumerate(shape):
                self.set_argument_i(self.shape[ix], d)
        return self.run(inputs, outputs)


class Permute(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Permute, self).__init__(key, ctx, **kwargs)
        self.len_perms = kwargs.get('len_perms', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.perms = [self.register_argument('perms[{}]'.format(i))
                for i in range(self.len_perms)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Transpose',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'perms_desc': [d for d in self.perms]
                    if len(self.perms) > 0 else None,
            }
        }

    def forward(self, x, perms):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        if perms is not None:
            for ix, d in enumerate(perms):
                self.set_argument_i(self.perms[ix], d)
        return self.run(inputs, outputs)


class Repeat(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Repeat, self).__init__(key, ctx, **kwargs)
        self.len_times = kwargs.get('len_times', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.times = [self.register_argument('times[{}]'.format(i))
                for i in range(self.len_times)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Tile',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'multiples_desc': [d for d in self.times]
                    if len(self.times) > 0 else None,
            }
        }

    def forward(self, x, times):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        if times is not None:
            for ix, d in enumerate(times):
                self.set_argument_i(self.times[ix], d)
        return self.run(inputs, outputs)