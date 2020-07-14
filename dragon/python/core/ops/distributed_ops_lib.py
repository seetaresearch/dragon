# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The distributed ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Collective(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Collective, self).__init__(key, dev, **kwargs)
        self.root = kwargs.get('root', 0)
        self.operation = kwargs.get('operation', 'MEAN')
        self.communication = kwargs.get('communication', None)
        self.group = kwargs.get('group', None)

    def attributes(self):
        arguments = self.group.arguments
        arguments['root'] = self.root
        arguments['operation'] = self.operation
        arguments['communication'] = self.communication
        return {'op_type': 'Collective', 'arguments': arguments}

    def forward(self, inputs):
        return self.dispatch(inputs, inputs, no_grad=True)
