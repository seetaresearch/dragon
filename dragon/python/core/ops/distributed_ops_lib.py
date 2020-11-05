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
"""Distributed ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Collective(Operator):
    """Collective operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Collective, self).__init__(key, dev, **kwargs)
        self.root = kwargs.get('root', 0)
        self.operation = kwargs.get('operation', 'MEAN')
        self.communication = kwargs.get('communication', None)
        self.group = kwargs.get('group', None)

    def attributes(self):
        """
        A dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        arguments = self.group.arguments
        arguments['root'] = self.root
        arguments['operation'] = self.operation
        arguments['communication'] = self.communication
        return {'op_type': 'Collective', 'arguments': arguments}

    def forward(self, inputs):
        """
        Run the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, inputs, no_grad=True)
