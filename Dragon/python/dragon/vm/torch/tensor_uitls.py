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

import importlib
import numpy as np
import dragon.core.tensor_utils as dg_tensor_utils


def from_numpy(data):
    """Create a tensor from the given numpy data.

    Parameters
    ----------
    data : numpy.ndarray
        The nd-array with various data type.

    Return
    ------
    vm.torch.Tensor
        The torch-based tensor.

    """
    if not isinstance(data, np.ndarray):
        raise TypeError('The data should be a numpy.ndarray.')
    if str(data.dtype) not in __NUMPY_TYPE_TO_TORCH:
        raise ValueError('Unsupported type({}) to torch tensor.'.format(data.dtype))
    module = importlib.import_module('dragon.vm.torch.tensor')
    return getattr(module, type_np2torch(str(data.dtype)))(data)


def to_numpy(tensor):
    """Create a numpy nd-array from the given tensor.

    Parameters
    ----------
    tensor : vm.torch.Tensor
        The tensor with various data type.

    Returns
    -------
    numpy.ndarray
        The numpy nd-array.

    """
    return tensor.numpy()


def from_dragon(tensor, own_storage=False):
    """Create a torch tensor from a existing dragon tensor.

    Set ``own_storage`` as ``True`` for automatically releasing the storage.

    Parameters
    ----------
    tensor : Tensor or str
        The dragon tensor.
    own_storage : boolean
        Whether to release storage during deconstructing.

    Returns
    -------
    vm.torch.Tensor
        The torch tensor.

    """
    info = dg_tensor_utils.GetTensorInfo(tensor)
    if not info['init']: return None
    module = importlib.import_module('dragon.vm.torch.tensor')
    th_tensor = getattr(module, type_np2torch(info['dtype']))()
    th_tensor._ctx = (info['mem_at'], info['device_id'])
    th_tensor._from_numpy = info['from_numpy']
    th_tensor._dg_tensor = tensor
    th_tensor._own_storage = own_storage
    return th_tensor


def to_str(tensor):
    """Return a format str representing the storage of a tensor.

    Parameters
    ----------
    tensor : vm.torch.Tensor
        The tensor with various data type.

    Returns
    -------
    str
        The format str.

    """
    return str(tensor)


def type_np2torch(type):
    return __NUMPY_TYPE_TO_TORCH[type]


__NUMPY_TYPE_TO_TORCH = {
    'float32': 'FloatTensor',
    'float64': 'DoubleTensor',
    'int32': 'IntTensor',
    'int64': 'LongTensor',
    'uint8': 'ByteTensor',
}