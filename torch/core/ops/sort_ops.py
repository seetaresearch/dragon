# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Sort ops."""

from dragon.vm.torch.core.autograd.function import Function


def argsort(input, dim=-1, descending=False):
    """Return the index of sorted elements along the given dimension.

    :attr:`dim` could be negative:

    ```python
    # A negative dimension is the last-k dimension
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    index1 = torch.argsort(x, dim=1)
    index2 = torch.argsort(x, dim=-1)  # Equivalent
    ```

    Use :attr:`descending` to sort in the descending order:

    ```python
    x = torch.tensor([1, 2, 3])
    index1 = torch.argsort(-x, descending=False)
    index2 = torch.argsort(x, descending=True)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional, default=-1
         The dimension to sort.
    descending : bool, optional, default=False
        Sort in the descending order or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return sort(input, dim, descending)[1]


def sort(input, dim=-1, descending=False, out=None):
    """Return the sorted elements along the given dimension.

    By default, the last dimension is chosen:

    ```python
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    value1, index1 = torch.sort(x)
    value2, index2 = torch.sort(x, dim=1)  # Equivalent
    ```

    Sort in the descending order if ``descending`` is ``True``:

    ```python
    x = torch.tensor([1, 2, 3])
    _, index1 = torch.sort(-x)
    _, index2 = torch.sort(x, descending=True)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional, default=-1
         The dimension to sort elements.
    descending : bool, optional, default=False
        Sort in the descending order or not.
    out : Sequence[dragon.vm.torch.Tensor], optional
        The optional output value and index.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    return Function.apply(
        "Sort",
        input.device,
        [input],
        outputs=out if out else [None, None],
        axis=dim,
        descending=descending,
    )


def topk(input, k, dim=-1, largest=True, sorted=True, out=None):
    """Return the top k-largest or k-smallest elements along the given dimension.

    :attr:`dim` could be negative:

    ```python
    # A negative dimension is the last-k dimension
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    value1, index1 = torch.topk(x, k=2, dim=1)
    value2, index2 = torch.topk(x, k=2, dim=-1)  # Equivalent
    ```

    If :attr:`largest` is ``False``, the k-smallest elements are returned:

    ```python
    x = torch.tensor([1, 2, 3])
    _, index1 = torch.topk(-x, 1, largest=True)
    _, index2 = torch.topk(x, 1, largest=False)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    k : int
        The number of top elements to select.
    dim : int, optional, default=-1
         The dimension to retrieve.
    largest : bool, optional
        Return largest or smallest elements.
    sorted : bool, optional
        Whether to return elements in the sorted order.
    out : Sequence[dragon.vm.torch.Tensor], optional
        The output value and index tensor.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    return Function.apply(
        "TopK",
        input.device,
        [input],
        outputs=out if out else (None, None),
        k=k,
        axis=dim,
        largest=largest,
        sorted=sorted,
    )
