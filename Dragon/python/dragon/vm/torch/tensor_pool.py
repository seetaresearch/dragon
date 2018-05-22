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

"""Like ThreadPool, we apply TensorPool to manage the reused tensors.

Tensors with the same scope in the pool will be reused by turns,
which speeds up the whole system by reducing the unnecessary deconstructing.

Heuristically, we have used 4 pools with different scopes:

* scope(leaf): A Pool to reuse leaf tensors.

* scope(numpy): A pool to reuse leaf tensors from numpy.

* scope(join): A pool to reuse RT(runtime) tensors required by forward-backward.

* scope(detach): A pool to reuse RT(runtime) tensors required by forward only.

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


class TensorPool(object):
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

    def get(self, scope='detach'):
        handle = self.get_handle(scope)
        return '[TPool]{}/tensor:{}'.format(scope, handle)

    def put_handle(self, scope, handle):
        self._scope2keys[scope].append(int(handle))

    def put(self, name):
        if '[TPool]' in name:
            scope, handle = name[7:].split('/tensor:')
            if not handle.isdigit():
                raise ValueError('Got illegal name: ' + name + '.\n'
                    'Excepted the format: [TPool]/scope/tensor:handle')
            self.put_handle(scope, handle)
            return True
        else: return False


# Define a global pool
TPool = TensorPool()