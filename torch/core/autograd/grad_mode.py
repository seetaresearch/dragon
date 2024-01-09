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
"""Autograd modes."""

from dragon.core.util import decorator
from dragon.core.util import tls


def is_grad_enabled():
    """Return if gradient calculation enabled."""
    return _GLOBAL_GRAD_ENABLED.mode


def _set_grad_enabled(mode=True):
    """Set the enabling of gradient calculation."""
    global _GLOBAL_GRAD_ENABLED
    _GLOBAL_GRAD_ENABLED.mode = mode


class enable_grad(decorator._DecoratorContextManager):
    """Context-manager to enable gradient calculation.

    Examples:

    ```python
    x = torch.ones(2, 3, requires_grad=True)
    with torch.no_grad():
        with torch.enable_grad():
            y = x + 1
    y.backward()
    ```

    """

    def __init__(self):
        """Create a ``enable_grad`` context manager."""
        self.prev = is_grad_enabled()

    def __enter__(self):
        _set_grad_enabled(True)

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)


class inference_mode(decorator._DecoratorContextManager):
    """Context-manager to enable or disable inference mode.

    Examples:

    ```python
    x = torch.ones(2, 3, requires_grad=True)
    with torch.inference_mode():
        y = x + 1
    y.backward()  # RuntimeError
    ```

    """

    def __init__(self, mode=True):
        """Create a ``inference_mode`` context manager."""
        self.mode = mode
        self.prev = is_grad_enabled()

    def __enter__(self):
        _set_grad_enabled(not self.mode)

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)


class no_grad(decorator._DecoratorContextManager):
    """Context-manager to disable gradient calculation.

    Examples:

    ```python
    x = torch.ones(2, 3, requires_grad=True)
    with torch.no_grad():
        y = x + 1
    y.backward()  # RuntimeError
    ```

    """

    def __init__(self):
        """Create a ``no_grad`` context manager."""
        self.prev = is_grad_enabled()

    def __enter__(self):
        _set_grad_enabled(False)

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)


class set_grad_enabled(object):
    """Context-manager to set gradient calculation on or off.

    Examples:

    ```python
    x = torch.ones(2, 3, requires_grad=True)
    with torch.set_grad_enabled(mode=False):
        y = x + 1
    y.backward()  # RuntimeError
    ```

    """

    def __init__(self, mode):
        """Create a ``set_grad_enabled`` context manager.

        Parameters
        ----------
        mode : bool
            Whether to enable calculation.

        """
        self.prev = is_grad_enabled()
        _set_grad_enabled(mode)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)


_GLOBAL_GRAD_ENABLED = tls.Constant(mode=True)
