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
"""Training functions library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.autograd import function


class ParamUpdate(function.Function):
    """ParamUpdate function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ParamUpdate, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', '')
        self.op_handle = kwargs.get('op_handle', '')
        self.lr_mult = kwargs.get('lr_mult', 1)
        self.decay_mult = kwargs.get('decay_mult', 1)

    def attributes(self):
        """
        Returns the attributes of the attributes

        Args:
            self: (todo): write your description
        """
        return {
            'name': self.op_handle,
            'op_type': self.op_type,
            'arguments': {
                'lr_mult': float(self.lr_mult),
                'decay_mult': float(self.decay_mult),
            },
        }

    def forward(self, param, grad):
        """
        Perform a forward forward pass on the device.

        Args:
            self: (todo): write your description
            param: (todo): write your description
            grad: (todo): write your description
        """
        self._check_device([param, grad])
        return self.dispatch([grad], [param], no_grad=True)


class GradAccumulate(function.Function):
    """GradAccumulate function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(GradAccumulate, self).__init__(key, dev, **kwargs)
        self.momentum = kwargs.get('momentum', 1)

    def attributes(self):
        """
        Returns a dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Axpby',
            'arguments': {
                'alpha': 1.0,
                'beta': float(self.momentum),
            },
        }

    def forward(self, grad):
        """
        Perform a forward forward.

        Args:
            self: (todo): write your description
            grad: (todo): write your description
        """
        return self.dispatch([grad], [grad.id + '[accum]'], no_grad=True)
