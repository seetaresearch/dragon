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
"""Parameter wrapper class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    by ignoring the it's gradient:

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
            Whether to compute the gradient if necessary.

        """
        super(Parameter, self).__init__(
            device=tensor.device,
            impl=tensor._impl,
            requires_grad=requires_grad)
        self._is_leaf = True
        self._wrapped_tensor = tensor

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()
