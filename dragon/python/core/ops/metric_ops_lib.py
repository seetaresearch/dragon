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
"""Metric ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Metric(Operator):
    """Metric operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Metric, self).__init__(key, dev, **kwargs)
        self.reduction = kwargs.get('reduction', 'MEAN')

    def forward(self, inputs):
        """
        Run the module inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Accuracy(Metric):
    """Accuracy operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Accuracy, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.top_k = kwargs.get('top_k', 1)
        self.ignore_index = kwargs.get('ignore_index', None)

    def attributes(self):
        """
        Returns a dictionary of the axis.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Accuracy',
            'arguments': {
                'axis': self.axis,
                'top_k': self.top_k,
                'ignore_index': self.ignore_index,
            }
        }
