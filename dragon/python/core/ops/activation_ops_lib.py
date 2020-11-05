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
"""Activation ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Activation(Operator):
    """Base activation operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Activation, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')

    def attributes(self):
        """
        The attributes of the attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': self.op_type, 'arguments': {}}

    def forward(self, inputs, inplace=False):
        """
        Forward computation

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Dropout(Activation):
    """Dropout operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Dropout, self).__init__(key, dev, **kwargs)
        self.prob = kwargs.get('prob', 0.5)
        self.scale = kwargs.get('scale', True)

    def attributes(self):
        """
        A dict of attributes of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Dropout',
            'arguments': {
                'prob': self.prob,
                'scale': self.scale,
            }
        }


class DropBlock2d(Activation):
    """DropBlock2d operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(DropBlock2d, self).__init__(key, dev, **kwargs)
        self.block_size = kwargs.get('block_size', 7)
        self.keep_prob = kwargs.get('keep_prob', 0.9)
        self.alpha = kwargs.get('alpha', 1.)
        self.decrement = kwargs.get('decrement', 0.)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'DropBlock2d',
            'arguments': {
                'block_size': self.block_size,
                'keep_prob': self.keep_prob,
                'alpha': self.alpha,
                'decrement': self.decrement,
                'data_format': self.data_format,
            },
        }


class DropPath(Activation):
    """DropPath operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(DropPath, self).__init__(key, dev, **kwargs)
        self.prob = kwargs.get('prob', 0.2)
        self.increment = kwargs.get('increment', 0.)

    def attributes(self):
        """
        Returns a dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'DropPath',
            'arguments': {
                'prob': self.prob,
                'increment': self.increment,
            }
        }


class Elu(Activation):
    """Elu operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Elu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.)

    def attributes(self):
        """
        A dict of attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Elu',
            'arguments': {'alpha': float(self.alpha)},
        }


class HardSigmoid(Activation):
    """HardSigmoid operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(HardSigmoid, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 0.5)

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'HardSigmoid',
            'arguments': {
                'alpha': float(self.alpha),
                'beta': float(self.beta),
            },
        }


class HardSwish(Activation):
    """HardSwish operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(HardSwish, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 0.5)

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'HardSwish',
            'arguments': {
                'alpha': float(self.alpha),
                'beta': float(self.beta),
            },
        }


class PRelu(Operator):
    """PRelu operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(PRelu, self).__init__(key, dev, **kwargs)
        self.data_format = kwargs.get('data_format', 'NCHW')

    def attributes(self):
        """
        A dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'PRelu',
            'arguments': {'data_format': self.data_format},
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Relu(Activation):
    """Relu operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Relu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 0.)

    def attributes(self):
        """
        A dict of attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Relu',
            'arguments': {'alpha': float(self.alpha)},
        }


class Relu6(Activation):
    """Relu6 operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a devu6 device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Relu6, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Relu',
            'arguments': {'max_value': 6.},
        }


class Selu(Activation):
    """Selu operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a devu device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Selu, self).__init__(key, dev, **kwargs)
        self.alpha = kwargs.get('alpha', 1.67326)
        self.gamma = kwargs.get('gamma', 1.0507)

    def attributes(self):
        """
        A dictionary of the attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Selu',
            'arguments': {
                'alpha': float(self.alpha),
                'gamma': float(self.gamma),
            }
        }


class Softmax(Activation):
    """Softmax operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Softmax, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)

    def attributes(self):
        """
        Return the attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Softmax',
            'arguments': {'axis': self.axis},
        }
