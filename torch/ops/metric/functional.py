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

from dragon.vm.torch.ops.metric import _functions


def topk_acc(input, label, k, dim=None):
    """Compute the top-k accuracy according to the label.

    If ``dim`` is not given, the last dimension of the input is chosen.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    label : dragon.vm.torch.Tensor
        The sparse label.
    k : int, optional, default=1
        The top-K results to return.
    dim : int, optional
        The axis of tensor to compute reduce value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if dim is None:
        dim = input.ndimension() - 1
    return _functions.Accuracy \
        .instantiate(
            input.device,
            axis=dim,
            top_k=k,
        ).apply(input, label)
