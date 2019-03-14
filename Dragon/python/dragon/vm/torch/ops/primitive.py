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
from dragon.vm.torch.c_api import device as _Device


def UnifyDevices(tensors, key='Inputs'):
    types, indices = [t.device.type for t in tensors], [0]
    if len(set(types)) != 1:
        raise ValueError('{} from different device type: [{}].'
            .format(key, ', '.join(types)))
    if types[0] == 'cuda':
        indices = [t.device.index for t in tensors]
        if len(set(indices)) != 1:
            raise ValueError('{} from different cuda device: [{}].'
                .format(key, ', '.join([str(d) for d in indices])))
    return _Device(types[0], indices[0])


def MakeDevice(inputs=(), outputs=()):
    # Case #1: [], [] -> CPU
    # Case #2: [...], [] -> Refer Inputs
    # Case #3: [], [...] -> Refer Outputs
    # Case #4: [...], [...] -> Refer Outputs
    if len(outputs) > 0: return UnifyDevices(outputs, 'Outputs')
    if len(inputs) > 0: return UnifyDevices(inputs, 'Inputs')
    return _Device()


def WrapScalar(scalar, dtype, device):
    # We use (DType + Value) to hash different scalars
    # Setting a Tensor with same DType and shape will not deconstruct it
    if 'float' in dtype: scalar = float(scalar)
    if 'int' in dtype: scalar = int(scalar)
    name = '/share/scalar/{}/{}'.format(dtype, str(scalar))
    if not dg.workspace.HasTensor(name):
        dg.workspace.FeedTensor(name, np.array(scalar, dtype=dtype))
    t = Tensor(name=name, dtype=dtype, device=device, own_storage=False)
    t.requires_grad = False
    return t
