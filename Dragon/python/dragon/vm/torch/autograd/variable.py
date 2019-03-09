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

import warnings

import dragon.core.tensor_utils as tensor_utils
import dragon.core.workspace as ws

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.pool import TensorPool, OperatorPool


def Variable(tensor, requires_grad=False, volatile=False):
    if volatile:
        warnings.warn("volatile was removed and now has no effect. "
                      "Use `with torch.no_grad():` instead.", stacklevel=2)
    if requires_grad and volatile:
        raise RuntimeError("Variable can't be volatile and require_grad at the same time!")
    tensor._requires_grad = requires_grad
    if requires_grad: tensor._ignored_grads = None
    return tensor


@property
def volatile(self):
    warnings.warn("volatile was removed (Variable.volatile is always False)", stacklevel=2)
    return False


def backward(self, gradient=None):
    if not self._requires_grad:
        raise RuntimeError('This variable does not require grads.'
                           '\nCan not backward from this variable.')

    # 1. Expressions -> Forward-Ops
    # We should sort out the topology of these operators before using
    all_expressions = sorted(self.__jit_recorder__.ops.items(), key=lambda d: d[0])
    forward_ops = [v for k, v in all_expressions]

    # 2. Forward-Ops + Targets + InputGrads + IgnoredGrads -> Backward-Ops
    targets = [self.name]; input_grads = []
    ignored_grads = list(self._ignored_grads) if self._ignored_grads else []
    if gradient is not None:
        if not isinstance(gradient, Tensor):
            raise TypeError('gradients can be either Tensors, Variables or None,'
                            ' but got {}'.format(type(gradient)))
        tensor_utils.FromPyArray(gradient.cpu().numpy(), self.name + '_grad')
        input_grads.append(self.name + '_grad')

    # 3. Flow or Flow or Flow
    ws.FlowGradients(forward_ops, targets, input_grads, ignored_grads)

    # 4. Release resources
    # We should release both the operator handles and tensors
    for forward_op in forward_ops:
        OperatorPool.put(forward_op.name)
        for output in forward_op.output:
            if output not in forward_op.input:
                TensorPool.put(output)


Tensor.backward = backward
Tensor.volatile = volatile