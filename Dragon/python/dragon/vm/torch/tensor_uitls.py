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

import dragon.core.mapping as mapping
from dragon.core.tensor_utils import GetTensorInfo


def from_numpy(data):
    """Create a tensor from the given numpy array.

    Parameters
    ----------
    data : ndarray
        The array with various data type.

    Return
    ------
    dragon.vm.torch.Tensor
        The torch tensor.

    """
    if not isinstance(data, np.ndarray):
        raise TypeError('The data should be a numpy.ndarray.')
    if str(data.dtype) not in mapping.TENSOR_TYPE_TO_TORCH_TENSOR:
        raise ValueError('Unsupported type({}) to torch tensor.'.format(data.dtype))
    module = importlib.import_module('dragon.vm.torch.tensor')
    return getattr(module, mapping.TENSOR_TYPE_TO_TORCH_TENSOR[str(data.dtype)])(data)


def to_numpy(tensor):
    """Create a numpy nd-array from the given tensor.

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The tensor with various data type.

    Returns
    -------
    numpy.ndarray
        The numpy array.

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
    dragon.vm.torch.Tensor
        The torch tensor.

    """
    info = GetTensorInfo(tensor)
    if not info or not info['init']: return None
    module = importlib.import_module('dragon.vm.torch.tensor')
    th_tensor = getattr(module, mapping.TENSOR_TYPE_TO_TORCH_TENSOR[info['dtype']])()
    th_tensor._ctx = (info['mem_at'], info['device_id'])
    th_tensor._from_numpy = info['from_numpy']
    th_tensor._dg_tensor = tensor
    th_tensor._own_storage = own_storage
    return th_tensor


def to_str(tensor):
    """Return a format str representing the storage of a tensor.

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The tensor with various data type.

    Returns
    -------
    str
        The format str.

    """
    return str(tensor)