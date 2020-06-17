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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from dragon.core.framework import workspace
from dragon.vm.torch.tensor import Tensor


def Variable(tensor, requires_grad=False, volatile=False):
    if volatile:
        warnings.warn("volatile was removed and now has no effect. "
                      "Use `with torch.no_grad():` instead.", stacklevel=2)
    if requires_grad and volatile:
        raise RuntimeError("Variable can't be volatile and require_grad at the same time!")
    tensor.requires_grad = requires_grad
    return tensor


@property
def volatile(self):
    warnings.warn("volatile was removed (Variable.volatile is always False)", stacklevel=2)
    return False


def backward(self, gradient=None):
    if not self._requires_grad:
        raise RuntimeError(
            'This variable does not require grads.'
            '\nCan not backward from this variable.'
        )

    # Collect and sort out the operation from tapes.
    operations = [v for k, v in sorted(self.__tape__.operations.items())]

    # Prepare resources to optimize the backward pass.
    input_grads = []
    if gradient is not None:
        if not isinstance(gradient, Tensor):
            raise TypeError(
                '<gradient> can be either Tensor, Variable or None, '
                'got {}'.format(type(gradient).__name__)
            )
        if gradient.shape != self.shape:
            raise ValueError(
                'Except the dimensions of <gradient> is {}, '
                'got {}.'.format(self.shape, gradient.shape))
        input_grads.append(gradient.id)

    # Dispatch the backward execution.
    workspace.run_backward(
        operations,
        targets=[self.id],
        sources=None,
        input_grads=input_grads,
        ignored_grads=list(self._ignored_grads),
    )

    # Release the holt resources.
    gc = workspace.get_workspace().collectors

    for op_def in operations:
        gc.OPERATOR.collect(op_def.name)
        for output in op_def.output:
            if output not in op_def.input:
                gc.TENSOR.collect(output)


# The monkey-patching.
Tensor.backward = backward
Tensor.volatile = volatile
