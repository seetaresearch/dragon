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
"""Sort ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


@OpSchema.num_inputs(1)
def argsort(inputs, axis=-1, descending=False, **kwargs):
    """Return the index of sorted elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    x = dragon.constant([[1, 2, 3], [3, 2, 1]])
    index1 = dragon.argsort(x, axis=1)
    index2 = dragon.argsort(x, axis=-1)  # Equivalent
    ```

    Use :attr:`descending` to sort in the inverse order:

    ```python
    x = dragon.constant([1, 2, 3])
    index1 = dragon.argsort(-x, descending=False)
    index2 = dragon.argsort(x, descending=True)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to sort elements.
    descending : bool, optional, default=False
        Sort in the descending order or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Sort', inputs, outputs=[None, None], axis=axis,
            descending=descending)[1]
    return OpLib.add('Sort', inputs, axis=axis, descending=descending,
                     num_outputs=2, **kwargs)[1]


@OpSchema.num_inputs(1)
def sort(inputs, axis=-1, descending=False, **kwargs):
    """Return the sorted elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    # A negative axis is the last-k axis
    x = dragon.constant([[1, 2, 3], [3, 2, 1]])
    value1, index1 = dragon.sort(x, axis=1)
    value2, index2 = dragon.sort(x, axis=-1)  # Equivalent
    ```

    Use :attr:`descending` to sort in the inverse order:

    ```python
    x = dragon.constant([1, 2, 3])
    _, index1 = dragon.sort(-x, descending=False)
    _, index2 = dragon.sort(x, descending=True)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to sort.
    descending : bool, optional, default=False
        Sort in the descending order or not.

    Returns
    -------
    Sequence[dragon.Tensor]
        The value and index tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Sort', inputs, outputs=[None, None], axis=axis,
            descending=descending)
    return OpLib.add('Sort', inputs, axis=axis, descending=descending,
                     num_outputs=2, **kwargs)


@OpSchema.num_inputs(1)
def top_k(inputs, k=1, axis=-1, largest=True, sorted=True, **kwargs):
    """Return the top k-largest or k-smallest elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    # A negative axis is the last-k axis
    x = dragon.constant([[1, 2, 3], [3, 2, 1]])
    value1, index1 = dragon.math.top_k(x, k=2, axis=1)
    value2, index2 = dragon.math.top_k(x, k=2, axis=-1)  # Equivalent
    ```

    If ``largest`` is ``False``, the k-smallest elements are returned:

    ```python
    x = dragon.constant([1, 2, 3])
    _, index1 = dragon.math.top_k(-x, largest=True)
    _, index2 = dragon.math.top_k(x, largest=False)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    k : int, optional, default=1
        The number of top elements to select.
    axis : int, optional, default=-1
        The axis to retrieve.
    largest : bool, optional, default=True
        Return largest or smallest elements.
    sorted : bool, optional, default=True
        Whether to return elements in the sorted order.

    Returns
    -------
    Sequence[dragon.Tensor]
        The value and index tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'TopK', inputs, outputs=[None, None], k=k, axis=axis,
            largest=largest, sorted=sorted)
    return OpLib.add('TopK', inputs, num_outputs=2, k=k, axis=axis,
                     largest=largest, sorted=sorted, **kwargs)
