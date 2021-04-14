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
from dragon.vm.torch.core import cpp
from dragon.vm.torch.core.tensor import Tensor


def from_numpy(ndarray):
    """Create a tensor converting from the given numpy array.

    Parameters
    ----------
    ndarray : numpy.ndarray
        The numpy array data.

    Return
    ------
    dragon.vm.torch.Tensor
        The torch tensor.

    """
    if not isinstance(ndarray, numpy.ndarray):
        raise TypeError('<ndarray> should be a numpy array.')
    return Tensor(ndarray, copy=False)


def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a tensor initializing from the given data.

    Parameters
    ----------
    data : array_like
        The data to initialize from.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        ``True`` to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    array_data = numpy.array(data, copy=True)
    if dtype is None:
        dtype = str(array_data.dtype)
    else:
        array_data = array_data.astype(dtype)
    return Tensor(
        array_data,
        dtype=dtype,
        device=cpp.device() if device is None else device,
        requires_grad=requires_grad,
    )


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
