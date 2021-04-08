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
"""Constant ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import workspace
from dragon.vm.torch.core.tensor import Tensor


def remove_scalars(input1, input2):
    """Remove the input scalars."""
    if isinstance(input1, Tensor):
        return input1, get_scalar(input2, input1.dtype, input1.device)
    return get_scalar(input1, input2.dtype, input2.device), input2


def get_scalar(input, dtype, device):
    """Return a cached scalar."""
    if isinstance(input, Tensor):
        return input
    try:
        input = float(input)
    except (TypeError, ValueError):
        raise ValueError(
            '<input> should be a python number, got {}.'
            .format(type(input).__name__))
    cached_name = '%s(%s)' % (dtype, input)
    default_ws = workspace.get_workspace()
    impl = default_ws.get_tensor(cached_name)
    if impl is None:
        impl = default_ws.create_tensor(cached_name)
        impl.FromNumpy(numpy.array(input, dtype), True)
    return Tensor(device=device, impl=impl)
