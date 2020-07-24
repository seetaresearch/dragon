# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import context
from dragon.core.framework import types
from dragon.vm.tensorflow.core.framework import constant_op


def convert_to_tensor(
    value,
    dtype=None,
    name=None,
    preferred_dtype=None,
):
    """Converts the given value to a Tensor.

    Parameters
    ----------
    value : number, sequence or numpy.ndarray
        The value to convert.
    dtype : dragon.vm.tensorflow.dtypes.DType, optional
        The optional data type.
    name : str, optional
        The Optional name.
    preferred_dtype : dragon.vm.tensorflow.dtypes.DType, optional
        The optional type when ``dtype`` is *None*.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if types.is_tensor(value):
        return value
    return constant_op.constant(value, dtype=dtype, name=name)


def name_scope(name):
    """Context-manager to nest the name as prefix for operations.

    Examples:

    ```python
    with tf.name_scope('my_scope'):
        x = tf.constant(1)
    print(x.name)
    ```

    Parameters
    ----------
    name : str
        The prefix name.

    Returns
    -------
    str
        The current nesting prefix.

    """
    return context.name_scope(name)


def device(device_name):
    """Context-manager to nest the the device spec.

    Examples:

    ```python
    with tf.device('/gpu:0'):
        x = tf.constant(1)
    ```

    Parameters
    ----------
    device_name : str
        The device descriptor.

    Returns
    -------
    Dict
        The current default device spec.

    """
    if not isinstance(device_name, str):
        raise TypeError('The device function should be a str.')
    device_and_id = device_name.split('/')[1]
    device, id = device_and_id.split(':')
    if device not in ['cpu', 'gpu']:
        raise ValueError('The device should either be cpu or gpu.')
    try:
        id = int(id)
    except Exception:
        raise ValueError('The device id should be a integer.')
    return context.device(device, device_id=id)
