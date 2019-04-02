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

import copy
import numpy
import importlib

from dragon.core import mapping as _mapping
from dragon.core import workspace as _workspace
from dragon.core import tensor_utils as _tensor_utils


class Size(tuple):
    def __init__(self, sizes):
        super(Size, self).__init__()

    def __setitem__(self, key, value):
        raise TypeError("'torch.Size' object does not support item assignment")

    def __repr__(self):
        return 'torch.Size([{}])'.format(', '.join([str(s) for s in self]))


class device(object):
    def __init__(self, type='cpu', index=0):
        self.type, self.index = type, index

    def copy(self):
        return copy.deepcopy(self)

    def __eq__(self, other):
        return self.type == other.type and \
               self.index == other.index

    def __str__(self):
        return '{}:{}'.format(self.type, self.index)

    def __repr__(self):
        return 'device(type={}, index={})'.format(self.type, self.index)


def from_numpy(data):
    """Create a tensor from the given numpy array.

    Parameters
    ----------
    data :  numpy.ndarray
        The array with various data type.

    Return
    ------
    dragon.vm.torch.Tensor
        The torch tensor.

    """
    if not isinstance(data, numpy.ndarray):
        raise TypeError('The data should be a numpy.ndarray.')
    if str(data.dtype) not in _mapping.TENSOR_TYPE_TO_TORCH_TENSOR:
        raise ValueError('Unsupported type({}) to torch tensor.'.format(data.dtype))
    module = importlib.import_module('dragon.vm.torch.tensor')
    return getattr(module, _mapping.TENSOR_TYPE_TO_TORCH_TENSOR[str(data.dtype)])(data)


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
    storage = _tensor_utils.GetStorage(tensor)
    if storage is None: return None
    module = importlib.import_module('dragon.vm.torch.tensor')
    T = getattr(module, _mapping.TENSOR_TYPE_TO_TORCH_TENSOR[storage.dtype])()
    T._storage, T._own_storage, T._tensor = storage, own_storage, tensor
    T._device = device(*storage.device)
    return T


def _get_tensor_pool():
    """Return the tensor pool of current workspace."""
    return _workspace.get_default_workspace().tensor_pool


def _get_operator_pool():
    """Return the operator pool of current workspace."""
    return _workspace.get_default_workspace().operator_pool