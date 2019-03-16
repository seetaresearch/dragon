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


class Fundamental(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Fundamental, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', 'Add')
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': self.op_type, 'arguments': {}}

    def forward(self, x1, x2, y):
        inputs = [x1, x2]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Maximum(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Maximum, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Maximum', 'arguments': {}}

    def forward(self, x1, x2, y):
        inputs = [x1, x2]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Minimum(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Minimum, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Minimum', 'arguments': {}}

    def forward(self, x1, x2, y):
        inputs = [x1, x2]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Clamp(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Clamp, self).__init__(key, dev, **kwargs)
        self.min = kwargs.get('min', None)
        self.max = kwargs.get('max', None)
        if self.min is not None: self.min = float(self.min)
        if self.max is not None: self.max = float(self.max)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Clip',
            'arguments': {
                'low': self.min,
                'high': self.max,
            },
        }

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Log(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Log, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Log', 'arguments': {}}

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Exp(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Exp, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Exp', 'arguments': {}}

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Sqrt(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Sqrt, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Sqrt', 'arguments': {}}

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class MM(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(MM, self).__init__(key, dev, **kwargs)
        self.transA = kwargs.get('transA', False)
        self.transB = kwargs.get('transB', False)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Matmul',
            'arguments': {
                'transA': self.transA,
                'transB': self.transB,
            }}

    def forward(self, x1, x2, y):
        inputs = [x1, x2]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class FullyConnected(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(FullyConnected, self).__init__(key, dev, **kwargs)
        self.transW = kwargs.get('transW', True)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'FullyConnected',
            'arguments': {'transW': self.transW},
        }

    def forward(self, x, w, b=None, y=None):
        inputs = [x, w] + ([b] if b else [])
        self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)


class Accumulate(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Accumulate, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)
        self.beta = kwargs.get('beta', 1.)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Accumulate',
            'arguments': {
                'alpha': self.alpha,
                'beta': self.beta,
            },
        }

    def forward(self, x, y=None):
        outputs = [y] if y else [self.register_output()]
        return self.run([x], outputs, auto_grad=False)