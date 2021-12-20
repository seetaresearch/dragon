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
"""Autograd functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.tensor import Tensor


def backward(tensors, grad_tensors=None, retain_graph=False):
    """Compute the derivatives of tensors w.r.t. graph leaves.

    Parameters
    ----------
    tensors : Sequence[dragon.vm.torch.Tensor]
        The derivative targets.
    grad_tensors : Sequence[dragon.vm.torch.Tensor], optional
        The gradient of attr:`tensors`.
    retain_graph : bool, optional, default=False
        ``False`` to free the graph used to compute grad.

    """
    # Check outputs.
    for i, output in enumerate(tensors):
        if not output._requires_grad:
            raise ValueError('Element %d of tensors does not requires grad.' % i)

    # Check grad outputs.
    if grad_tensors is not None:
        if len(grad_tensors) != len(tensors):
            raise ValueError('Number of tensors and grad tensors should be same.')
        for i, grad_tensor in enumerate(grad_tensors):
            if not isinstance(grad_tensor, Tensor):
                raise TypeError(
                    'Element {} of grad tensors should be a tensor, got {}.'
                    .format(i, type(grad_tensor).__name__))
            if grad_tensor.shape != tensors[i].shape:
                raise ValueError(
                    'Size of element {} of grad tensors should be {}, got {}.'
                    .format(i, tensors[i].shape, grad_tensor.shape))
    else:
        grad_tensors = []

    return Function.backward(tensors, grad_tensors, retain_graph=retain_graph)
