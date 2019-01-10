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

"""Implement some resource pools based on the dummy name. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
from collections import defaultdict, deque


class _TensorPool(object):
    """We apply the TensorPool to manage the reused tensors.

    Tensors with the same scope in the pool will be reused by turns,
    which speeds up the whole system by reducing the unnecessary deconstructing.

    Heuristically, we have used 4 pools with different scopes:

    * scope(Leaf): A Pool to reuse leaf tensors.

    * scope(NumPy): A pool to reuse leaf tensors from numpy.

    * scope(Join): A pool to reuse RT(runtime) tensors required by forward-backward.

    * scope(Detach): A pool to reuse RT(runtime) tensors required by forward only.

    """
    def __init__(self):
        # deque provide much higher performance than Queue
        self._scope2keys = defaultdict(deque)

    def get(self, scope='${DETACH}'):
        try:
            return self._scope2keys[scope].popleft()
        except:
            self._scope2keys[scope].append(
                dragon.workspace.GetDummyName(
                    '${TORCH_TENSOR_POOL}/%s/Tensor' % scope,
                        domain='Tensor', zero_based=False))
            return self._scope2keys[scope].popleft()

    def put(self, name):
        if '${TORCH_TENSOR_POOL}' in name:
            scope, _ = name[21:].split('/')
            self._scope2keys[scope].append(name)
            return True
        else: return False


class _OperatorPool(object):
    """Operators whose gradients is required will hold a resource handle,
    which is also called ``Anchor`` in the backend.

    We apply this pool to collect the handles according to the type of operator,
    as the mem size of temporal resources varies greatly.

    The resource handle will be released after the gradient flow automatically.

    """
    def __init__(self):
        # deque provide much higher performance than Queue
        self._type2keys = defaultdict(deque)

    def get(self, op_type):
        try:
            return self._type2keys[op_type].popleft()
        except:
            self._type2keys[op_type].append(
                dragon.workspace.GetDummyName(
                    '${TORCH_OPS_POOL}/%s' % op_type,
                        domain='Operator', zero_based=False))
            return self._type2keys[op_type].popleft()

    def put(self, op_name):
        op_type, _ = op_name[18:].split('_')
        self._type2keys[op_type].append(op_name)


# Define the global pools
TensorPool = _TensorPool()
OperatorPool = _OperatorPool()