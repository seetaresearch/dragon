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

"""List some extended Tensor C++ API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core import workspace as _workspace
from dragon.core import proto_utils as _proto_utils
from dragon.core.tensor import Tensor as _Tensor


def FromShape(shape, dtype='float32', name=None):
    """Create a Tensor from the shape.

    If specifying a existed tensor with larger shape,
    the original tensor will be reset.

    Parameters
    ----------
    shape : tuple or list
        The shape info.
    dtype : str
        The data type.
    name : str, optional
        The optional tensor name.

    Returns
    -------
    Tensor
        The tensor with the specific shape.

    """
    tensor = _try_get_tensor(name)
    tensor.shape = list(shape)
    if not isinstance(shape, (tuple, list)):
        raise TypeError('The shape should be a tuple or list.')
    _get_workspace().TensorFromShape(
        _stringify_tensor(tensor), list(shape), dtype)
    return tensor


def SetShape(tensor, shape, dtype='float32'):
    """Set a Tensor with the shape.

    Parameters
    ----------
    tensor : Tensor or str, optional
        The specific tensor to use.
    shape : sequence of int
        The shape info.
    dtype : str, optional
        The data type.

    Returns
    -------
    None

    """
    _get_workspace().TensorFromShape(
        _stringify_tensor(tensor), shape, dtype)


def FromTensor(src, src_ctx=None, name=None, ctx=None):
    """Create a Tensor from a existing tensor.

    If specifying a existed tensor with larger shape,
    the original tensor will be reset.

    Parameters
    ----------
    src_ctx : str
        The name of source tensor.
    src_ctx : DeviceOption
        The context of source tensor.
    name : str
        The optional tensor name for destination tensor.
    ctx : DeviceOption
        The context for destination tensor.

    Returns
    -------
    Tensor
        The tensor with the same data as source.

    """
    tensor = _try_get_tensor(name)
    if src_ctx is None: src_ctx = _proto_utils.GetDeviceOption('cpu')
    if ctx is None: ctx = _proto_utils.GetDeviceOption('cpu')
    _get_workspace().TensorFromTensor(
        _stringify_tensor(tensor),
            _stringify_tensor(src),
                _stringify_proto(ctx),
                    _stringify_proto(src_ctx))
    return tensor


def FromArray(array, name=None):
    """Create a Tensor from a existing Array.

    Note that memory of Tensor are ``zero-copied``.

    If specifying a existed tensor with larger shape,
    the original tensor will be reset.

    Parameters
    ----------
    array : numpy.ndarray
        The array for creating the tensor.
    name : str
        The optional tensor name.

    Returns
    -------
    Tensor
        The tensor sharing the memory with original array.

    """
    tensor = _try_get_tensor(name)
    if not isinstance(array, numpy.ndarray):
        raise TypeError('Excepted a numpy.ndarray.')
    _get_workspace().TensorFromArray(
        _stringify_tensor(tensor), array)
    return tensor


def SetArray(tensor, array):
    """Set a Tensor from a existing Array.

    Note that memory of Tensor are ``zero-copied``.

    Parameters
    ----------
    tensor : Tensor or str, required
        The specific tensor to use.
    array : numpy.ndarray
        The array for creating the tensor.

    Returns
    -------
    None

    """
    _get_workspace().TensorFromArray(
        _stringify_tensor(tensor), array)


def ToArray(tensor, readonly=False):
    """Create a Array from a existing Tensor.

    Note that memory of Array are *zero-copied*.

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.
    readonly : boolean
        Whether to sync the contents with device.

    Returns
    -------
    numpy.ndarray
        The array sharing the memory with original tensor.

    """
    return  _get_workspace().TensorToArray(
        _stringify_tensor(tensor), readonly)


def GetStorage(tensor):
    """Get the storage of a existing Tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.

    Returns
    -------
    TensorStorage
        The storage of the backend.

    """
    tensor = _stringify_tensor(tensor)
    if not _get_workspace().HasTensor(tensor): return None
    return _get_workspace().GetTensor(tensor)


def _stringify_proto(obj):
    """Try to stringify a proto-buffer structure."""
    return obj.SerializeToString()


def _stringify_tensor(obj):
    """Try to stringify a tensor."""
    if hasattr(obj, 'name'): return obj.name
    else: return str(obj)


def _try_get_tensor(name=None):
    """Try to create or get a tensor"""
    if name is None or name == '': return _Tensor()
    else: return _Tensor.Ref(name)


def _get_workspace():
    """Get the current default workspace."""
    return _workspace.get_default_workspace()