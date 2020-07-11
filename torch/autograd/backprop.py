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

"""Do back-propagation from the executed functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch import executor


def backward(tensors, grad_tensors=None, retain_graph=False):
    """Compute the derivatives of tensors w.r.t. graph leaves.

    Parameters
    ----------
    tensors : Sequence[dragon.vm.torch.Tensor]
        The derivative targets.
    grad_tensors : Sequence[dragon.vm.torch.Tensor], optional
        The optional gradient of ``tensors``.
    retain_graph : bool, optional, default=False
        **False** to free the graph used to compute grad.

    """
    return executor.run_backward(
        tensors=tensors,
        grad_tensors=grad_tensors,
        retain_graph=retain_graph,
    )
