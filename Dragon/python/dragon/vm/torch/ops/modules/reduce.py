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


class Reduce(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Reduce, self).__init__(key, ctx, **kwargs)
        self.operation = kwargs.get('operation', 'SUM')
        self.dim = kwargs.get('dim', None)
        self.keepdim = kwargs.get('keepdim', True)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for reduce op.

        Mutable ``axis`` and ``keep_dims`` is non-trivial for backend,
        we simply hash them in the persistent key.

        """
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'Reduce',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'operation': self.operation,
                'axes': [self.dim] if self.dim is not None else None,
                'keep_dims': self.keepdim,
            }
        }

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output(x.dtype)]
        return self.run(inputs, outputs)


class ArgReduce(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(ArgReduce, self).__init__(key, ctx, **kwargs)
        self.operation = kwargs.get('operation', 'ARGMAX')
        self.axis = kwargs.get('axis', None)
        self.keepdim = kwargs.get('keepdim', True)
        self.top_k = kwargs.get('top_k', 1)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for reduce op.

        Mutable ``axis`` and ``keep_dims`` is non-trivial for backend,
        we simply hash them in the persistent key.

        """
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'ArgReduce',
            'n_inputs': 1, 'n_outputs': 2,
            'arguments': {
                'operation': self.operation if 'ARG' in self.operation \
                    else 'ARG' + self.operation,
                'axis': self.axis if self.axis else 2147483647,
                'keep_dims': self.keepdim,
                'top_k': self.top_k,
            }
        }

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        if 'ARG' in self.operation:
            # Return indices only
            outputs = [y] if y else [self.register_output(dtype='int64')]
            outputs += [self.register_output(x.dtype)]
            returns = self.run(inputs, outputs)
            return returns[0]
        else:
            if y:
                if not isinstance(y, (tuple, list)):
                    raise TypeError('Excepted outputs as a tuple or list, got {}.'.format(type(y)))
                if len(y) != 2:
                    raise ValueError('Excepted 2 outputs, got {}.'.format(len(y)))
                outputs = [y[1], y[0]]
            else: outputs = [self.register_output('int64'), self.register_output(x.dtype)]
            returns = self.run(inputs, outputs)
            # Return values only
            if self.axis is None: return returns[1]
            # Return values and indices
            return returns[1], returns[0]