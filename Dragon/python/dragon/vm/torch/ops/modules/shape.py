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
from dragon.vm.torch.tensor import ReferenceTensor


class Fill(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Fill, self).__init__(key, ctx, **kwargs)
        self.n_dim = kwargs.get('n_dim', 0)
        self.value = kwargs.get('value', 0.0)
        self.dtype = kwargs.get('dtype', 'float32')
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.shape = [self.register_argument('shape[{}]'.format(i))
                for i in range(self.n_dim)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Fill',
            'n_inputs': 0, 'n_outputs': 1,
            'arguments': {
                'dtype': self.dtype,
                'value': float(self.value),
                'dims_desc': [d for d in self.shape] if self.n_dim > 0 else None,
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
        self.n_dim = kwargs.get('n_dim', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.dims = [self.register_argument('dims[{}]'.format(i))
                for i in range(self.n_dim)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Reshape',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'dims_desc': [d for d in self.dims]
                    if self.n_dim > 0 else None,
            }
        }

    def forward(self, x, shape):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [ReferenceTensor(x)]
        if shape is not None:
            for ix, d in enumerate(shape):
                self.set_argument_i(self.dims[ix], d)
        return self.run(inputs, outputs)


class Squeeze(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Squeeze, self).__init__(key, ctx, **kwargs)
        self.dim = kwargs.get('dim', None)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for squeeze op."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'Squeeze',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {'axis': self.dim}
        }

    def forward(self, x, out=None):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [out] if out else [ReferenceTensor(x)]
        return self.run(inputs, outputs)


class UnSqueeze(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(UnSqueeze, self).__init__(key, ctx, **kwargs)
        self.dim = kwargs.get('dim', None)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for squeeze op."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'ExpandDims',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {'axis': self.dim}
        }

    def forward(self, x, out=None):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [out] if out else [ReferenceTensor(x)]
        return self.run(inputs, outputs)


class Permute(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Permute, self).__init__(key, ctx, **kwargs)
        self.n_perm = kwargs.get('n_perm', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.perm = [self.register_argument('perm[{}]'.format(i))
                for i in range(self.n_perm)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Transpose',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'perm_desc': [axis for axis in self.perm]
                    if self.n_perm > 0 else None,
            }
        }

    def forward(self, x, perm):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        if perm is not None:
            for ix, d in enumerate(perm):
                self.set_argument_i(self.perm[ix], d)
        return self.run(inputs, outputs)


class Repeat(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Repeat, self).__init__(key, ctx, **kwargs)
        self.n_times = kwargs.get('n_times', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.times = [self.register_argument('times[{}]'.format(i))
                for i in range(self.n_times)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Tile',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'multiples_desc': [n for n in self.times]
                    if self.n_times > 0 else None,
            }
        }

    def forward(self, x, times):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        if times is not None:
            for ix, d in enumerate(times):
                self.set_argument_i(self.times[ix], d)
        return self.run(inputs, outputs)