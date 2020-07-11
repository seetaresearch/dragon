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

import numpy

from dragon.core.framework import workspace
from dragon.vm.torch import cpp
from dragon.vm.torch.tensor import Tensor


def new_leaf(sizes, kwargs):
    """Return a leaf tensor from optional kwargs."""
    device = kwargs.get('device', cpp.device())
    return Tensor(
        *sizes,
        dtype=kwargs.get('dtype', 'float32'),
        device=cpp.device() if device is None else device,
        requires_grad=kwargs.get('requires_grad', False)
    )


def remove_binary_scalar(input, value):
    """Remove the python scalar for binary ops."""
    if isinstance(input, Tensor):
        return input, scalar_to_tensor(value, input.dtype, input.device)
    else:
        return scalar_to_tensor(input, value.dtype, value.device), value


def scalar_to_tensor(input, dtype, device):
    """Return a cached scalar tensor."""
    if isinstance(input, Tensor):
        return input
    try:
        input = float(input)
    except (TypeError, ValueError):
        raise ValueError(
            '<input> should be a python number, got {}.'
            .format(type(input).__name__)
        )
    name = '/share/scalar/{}/{}'.format(dtype, str(input))
    current_ws = workspace.get_workspace()
    if not current_ws.has_tensor(name):
        current_ws.feed_tensor(name, numpy.array(input, dtype=dtype))
    return Tensor(device=device, impl=current_ws.GetTensor(name), requires_grad=False)


def unify_devices(tensors, key='Inputs'):
    """Return a device unified from tensors."""
    types, indices = [t._device.type for t in tensors], [0]
    if len(set(types)) != 1:
        raise ValueError(
            '{} from different device type: [{}].'
            .format(key, ', '.join(types)))
    if types[0] == 'cuda':
        indices = [t._device.index for t in tensors]
        if len(set(indices)) != 1:
            raise ValueError(
                '{} from different cuda device: [{}].'
                .format(key, ', '.join([str(d) for d in indices])))
    return cpp.device(types[0], indices[0])
