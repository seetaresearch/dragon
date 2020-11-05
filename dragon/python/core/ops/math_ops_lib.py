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
"""Math ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Axpby(Operator):
    """Axpby operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Axpby, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)
        self.beta = kwargs.get('beta', 1.)

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Axpby',
            'arguments': {
                'alpha': self.alpha,
                'beta': self.beta,
            }
        }

    def forward(self, inputs, outputs=None):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            outputs: (todo): write your description
        """
        if outputs is None:
            outputs = [None] * len(inputs)
        outputs = [self.alloc(out) for out in outputs]
        return self.dispatch(inputs, outputs, no_grad=True)


class BinaryOp(Operator):
    """Binary operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Wrapper around the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(BinaryOp, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        """
        The attributes of the attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs, outputs=(None,)):
        """
        R callable.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            outputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc(outputs[0])])


class Clip(Operator):
    """Clip operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a low - level device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Clip, self).__init__(key, dev, **kwargs)
        self.low = kwargs.get('low', None)
        self.high = kwargs.get('high', None)
        if self.low is not None:
            self.low = float(self.low)
        if self.high is not None:
            self.high = float(self.high)

    def attributes(self):
        """
        Returns the attributes dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Clip',
            'arguments': {
                'low': self.low,
                'high': self.high,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class FullyConnected(Operator):
    """FullyConnected operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(FullyConnected, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.transpose_w = kwargs.get('transpose_w', True)

    def attributes(self):
        """
        A dict of the axis attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'FullyConnected',
            'arguments': {
                'axis': self.axis,
                'transW': self.transpose_w,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class MatMul(Operator):
    """MatMul operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the underlying device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(MatMul, self).__init__(key, dev, **kwargs)
        self.transpose_a = kwargs.get('transpose_a', False)
        self.transpose_b = kwargs.get('transpose_b', False)

    def attributes(self):
        """
        A copy of all attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'MatMul',
            'arguments': {
                'transA': self.transpose_a,
                'transB': self.transpose_b,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class UnaryOp(Operator):
    """Unary operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Wrapper around the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(UnaryOp, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        """
        The attributes of the attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])
