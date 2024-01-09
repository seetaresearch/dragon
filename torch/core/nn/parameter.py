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
"""NN Parameter."""

from dragon.vm.torch.core.tensor import Tensor


class Parameter(Tensor):
    r"""A wrapped tensor considered to be a module parameter.

    Use this class to wrap a leaf tensor to be a parameter,
    that can be identified by ``torch.nn.Module``:

    ```python
    param = torch.nn.Parameter(torch.ones(2, 3))
    ```

    Typically, the gradient of a parameter should be computed,
    while you can set ``requires_grad`` to ``False`` to ignore.
    Froze a parameter from updating can be directly implemented
    by ignoring it's gradient:

    ```python
    param = torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
    ```

    """

    def __init__(self, tensor, requires_grad=True):
        """Create a ``Parameter``.

        Parameters
        ----------
        tensor : dragon.vm.torch.Tensor
            The tensor to be wrapped.
        requires_grad : bool, optional, default=True
            Whether to compute gradient if necessary.

        """
        super(Parameter, self).__init__(
            device=tensor.device, impl=tensor._impl, requires_grad=requires_grad
        )
        self._is_leaf = True
        self._wrapped_tensor = tensor

    def __repr__(self):
        """Return the representation string.

        Returns
        -------
        str
            The representation string.

        """
        return "Parameter containing:\n" + super(Parameter, self).__repr__()

    def __setstate__(self, state):
        """Set the serialization state.

        Parameters
        ----------
        state : Dict
            The state dict.

        """
        self._is_leaf = True
        self._wrapped_tensor = Tensor()
        self._wrapped_tensor.__setstate__(state)
        state.pop("array", None)
        super(Parameter, self).__setstate__(state)
        self._impl = self._wrapped_tensor._impl
        self._deleter = None
