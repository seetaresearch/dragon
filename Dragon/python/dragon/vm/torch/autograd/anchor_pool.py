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

"""Like TensorPool, we apply AnchorPool to manage the reused anchors.

Anchors for the same operator will be reused by turns,
which stable the memory-reusing of temporal resources.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, deque


##############################################
#                                            #
#                Pool-Core                   #
#                                            #
##############################################


class AnchorPool(object):
    def __init__(self):
        self._scope2handle = defaultdict(int)
        # deque provide much higher performance that Queue
        self._scope2keys = defaultdict(deque)

    def get_handle(self, scope):
        if len(self._scope2keys[scope]) == 0:
            self._scope2handle[scope] += 1
            self._scope2keys[scope].append(
                self._scope2handle[scope] - 1)
        return self._scope2keys[scope].popleft()

    def put_handle(self, scope, handle):
        self._scope2keys[scope].append(int(handle))

    def get(self, op_type):
        handle = self.get_handle(op_type)
        return '{}:{}'.format(op_type, handle)

    def put(self, anchor):
        scope, handle = anchor.split(':')
        if not handle.isdigit():
            raise ValueError('Got illegal name: ' + anchor + '.\n'
                             'Excepted the format: OpType:handle')
        self.put_handle(scope, handle)


# Define a global pool
APool = AnchorPool()