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

from dragon.vm.torch.tensor import *


def CheckDataType(inputs, dtypes=None):
    if isinstance(inputs, Tensor): inputs = [inputs]
    if not isinstance(dtypes, (tuple, list)): dtypes = [dtypes]
    request_type = inputs[0]._dtype
    for ix, input in enumerate(inputs):
        if dtypes is not None and \
            input._dtype not in dtypes:
                raise TypeError('Type of input({}) is {}, '
                                'not in the support set: ({}).'
                                .format(ix, input._dtype, ', '.join(dtypes)))
        if input._dtype != request_type:
            raise TypeError('Excepted the type of input({}) is {}, got {}.'
                            .format(ix, request_type, input._dtype))


def UnifyDevices(tensors, key='Inputs'):
    device_types = [t._ctx[0] for t in tensors]
    devices = [0]
    if len(set(device_types)) != 1:
        raise ValueError('{} from different device type: [{}].'
                         .format(key, ', '.join(device_types)))
    if device_types[0] == 'CUDA':
        devices = [t._ctx[1] for t in tensors]
        if len(set(devices)) != 1:
            raise ValueError('{} from different cuda device: [{}].'
                             .format(key, ', '.join([str(d) for d in devices])))
    return device_types[0], devices[0]


def MakeContext(inputs=(), outputs=(), meta=None):
    type = 'CPU'; device_id = 0
    if len(inputs) > 0:
        type, device_id = UnifyDevices(inputs, 'Inputs')
    if len(outputs) > 0:
        type, device_id = UnifyDevices(outputs, 'Outputs')
    if meta is not None: type, device_id = meta
    # Case #1: [], [] -> CPU
    # Case #2: [...], [] -> Refer Inputs
    # Case #3: [], [...] -> Refer Outputs
    # Case #4: [...], [...] -> Refer Outputs
    # Case #5: meta -> CPU, CUDA:?
    return type, device_id