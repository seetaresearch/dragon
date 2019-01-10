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
from google.protobuf.message import Message

import dragon.import_c_api as C

from dragon.core.tensor import Tensor
from dragon.core.proto_utils import GetDeviceOption


__all__ = [
    'FromShape',
    'SetShape',
    'FromTensor',
    'FromPyArray',
    'SetPyArray',
    'ToPyArray',
    'ToPyArrayEx',
    'ToCPUTensor',
    'ToCUDATensor',
    'GetTensorInfo',
]


def FromShape(shape, dtype='float32', ctx=None, name=None):
    """Create a Tensor from the shape.

    If specifying a existed tensor with larger shape,
    the original tensor will be reset.

    Parameters
    ----------
    shape : tuple or list
        The shape info.
    dtype : str
        The data type.
    ctx : dragon_pb2.DeviceOption
        The context info.
    name : str, optional
        The optional tensor name.

    Returns
    -------
    Tensor
        The tensor with the specific shape.

    References
    ----------
    The wrapper of ``TensorFromShapeCC``.

    """
    tensor = _try_get_tensor(name)
    if not isinstance(shape, (tuple, list)):
        raise TypeError('The shape should be a tuple or list.')
    if ctx is None: ctx = GetDeviceOption('CPU')
    C.TensorFromShapeCC(
        _stringify_tensor(tensor),
        list(shape), dtype,
        _stringify_proto(ctx))
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

    References
    ----------
    The wrapper of ``TensorFromShapeCC``.

    """
    C.TensorFromShapeCC(_stringify_tensor(tensor), shape, dtype)


def FromTensor(src, src_ctx=None, name=None, ctx=None):
    """Create a Tensor from a existing tensor.

    If specifying a existed tensor with larger shape,
    the original tensor will be reset.

    Parameters
    ----------
    src_ctx : str
        The name of source tensor.
    src_ctx : dragon_pb2.DeviceOption
        The context of source tensor.
    name : str
        The optional tensor name for destination tensor.
    ctx : dragon_pb2.DeviceOption
        The context for destination tensor.

    Returns
    -------
    Tensor
        The tensor with the same data as source.

    References
    ----------
    The wrapper of ``TensorFromTensorCC``.

    """
    tensor = _try_get_tensor(name)
    if src_ctx is None: src_ctx = GetDeviceOption('CPU')
    if ctx is None: ctx = GetDeviceOption('CPU')
    C.TensorFromTensorCC(
        _stringify_tensor(tensor), _stringify_tensor(src),
        _stringify_proto(ctx), _stringify_proto(src_ctx))
    return tensor


def FromPyArray(array, name=None):
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

    References
    ----------
    The wrapper of ``TensorFromPyArrayCC``.

    """
    tensor = _try_get_tensor(name)
    if not isinstance(array, np.ndarray):
        raise TypeError('The given nd-array should be numpy.ndarray.')
    C.TensorFromPyArrayCC(_stringify_tensor(tensor), array)
    return tensor


def SetPyArray(tensor, array):
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

    References
    ----------
    The wrapper of ``TensorFromPyArrayCC``.

    """
    C.TensorFromPyArrayCC(_stringify_tensor(tensor), array)


def ToPyArray(tensor):
    """Create a Array from a existing Tensor.

    Note that memory of Array are ``zero-copied``.

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.

    Returns
    -------
    numpy.ndarray
        The array sharing the memory with original tensor.

    References
    ----------
    The wrapper of ``TensorToPyArrayCC``.

    """
    return C.TensorToPyArrayCC(_stringify_tensor(tensor))


def ToPyArrayEx(tensor):
    """Create a const Array from a existing Tensor.

    Note that memory of Array are ``zero-copied`` and ``const``.

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.

    Returns
    -------
    numpy.ndarray
        The array sharing the memory with original tensor.

    References
    ----------
    The wrapper of ``TensorToPyArrayExCC``.

    """
    return C.TensorToPyArrayExCC(_stringify_tensor(tensor))


def ToCPUTensor(tensor):
    """Switch the storage of a existing Tensor on cpu memory.

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.

    Returns
    -------
    None

    References
    ----------
    The wrapper of ``ToCPUTensorCC``.

    """
    return C.ToCPUTensorCC(_stringify_tensor(tensor))


def ToCUDATensor(tensor, device=0):
    """Switch the storage of a existing Tensor on cuda memory.

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.
    device : int
        The id of the device to use.

    Returns
    -------
    None

    References
    ----------
    The wrapper of ``ToCUDATensorCC``.

    """
    return C.ToCUDATensorCC(_stringify_tensor(tensor), device)


def GetTensorInfo(tensor, stream=1):
    """Get the info of a existing Tensor.

    The string info contains following fields:

    stream #1: ``dtype``, ``from_numpy``, ``init``, ``mem``, ``mem_at``, ``device_id``

    stream #2: ``shape``

    stream #3: #1 + #2

    Parameters
    ----------
    tensor : Tensor or str
        The input tensor.
    stream : int
        The stream id.

    Returns
    -------
    dict
        The info.

    References
    ----------
    The wrapper of ``GetTensorInfoCC``.

    """
    if not dg.workspace.HasTensor(_stringify_tensor(tensor)): return None
    info = C.GetTensorInfoCC(_stringify_tensor(tensor), stream)
    info['mem'] = []
    if 'CPU' in info:
        info['mem'].append('CPU'); info['device_id'] = 0
    if 'CUDA' in info:
        info['mem'].append('CUDA'); info['device_id'] = int(info['CUDA'])
    if 'CNML' in info:
        info['mem'].append('CNML'); info['device_id'] = int(info['CNML'])
    info['init'] = len(info['mem']) > 0
    return info


def _stringify_proto(obj):
    """Try to stringify a proto-buffer structure."""
    if obj is str: return obj
    elif isinstance(obj, Message): return obj.SerializeToString()
    else: raise TypeError('Object can not be serialized as a string.')


def _stringify_tensor(obj):
    """Try to stringify a tensor."""
    if hasattr(obj, 'name'): return obj.name
    else:
        try:
            obj = str(obj)
        except Exception as e:
            raise TypeError('Object can bot be used as a tensor. Error: {0}'.format(str(e)))
        return obj


def _try_get_tensor(name=None):
    """Try to create or get a tensor"""
    if name is None or name == '':
        return Tensor()
    else:
        tensor = Tensor('')
        tensor.set_name(name)
        return tensor