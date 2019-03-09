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

import numpy as np
import dragon as dg

from dragon.vm.torch.tensor import *
from dragon.vm.torch.c_api import Context


def UnifyDevices(tensors, key='Inputs'):
    device_types = [t._ctx.device_type for t in tensors]
    device_ids = [0]
    if len(set(device_types)) != 1:
        raise ValueError('{} from different device type: [{}].'
            .format(key, ', '.join(device_types)))
    if device_types[0] == 'CUDA':
        device_ids = [t._ctx.device_id for t in tensors]
        if len(set(device_ids)) != 1:
            raise ValueError('{} from different cuda device: [{}].'
            .format(key, ', '.join([str(d) for d in device_ids])))
    return Context(device_types[0], device_ids[0])


def MakeContext(inputs=(), outputs=()):
    # Case #1: [], [] -> CPU
    # Case #2: [...], [] -> Refer Inputs
    # Case #3: [], [...] -> Refer Outputs
    # Case #4: [...], [...] -> Refer Outputs
    if len(outputs) > 0: return UnifyDevices(outputs, 'Outputs')
    if len(inputs) > 0: return UnifyDevices(inputs, 'Inputs')
    return Context()


def WrapScalar(scalar, dtype, ctx):
    # We use (DType + Value) to hash different scalars
    # Setting a Tensor with same DType and shape will not deconstruct it
    if 'float' in dtype: scalar = float(scalar)
    if 'int' in dtype: scalar = int(scalar)
    name = '/share/scalar/{}/{}'.format(dtype, str(scalar))
    if not dg.workspace.HasTensor(name):
        dg.workspace.FeedTensor(name, np.array(scalar, dtype=dtype))
    t = Tensor(name=name, dtype=dtype, ctx=ctx, own_storage=False)
    t.requires_grad = False
    return t
