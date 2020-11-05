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
#     <https://github.com/pytorch/pytorch/blob/master/torch/autograd/grad_mode.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import tls


class enable_grad(object):
    """Context-manager to enable gradient calculation.

    Examples:

    ```python
    x = torch.ones(2, 3, requires_grad=True)
    with torch.no_grad():
        with torch.enable_grad():
            y = x + 1
    y.backward()  # 0 error(s), 0 warning(s)
    ```

    """

    def __init__(self):
        """Create a ``enable_grad`` context manager."""
        self.prev = is_grad_enabled()

    def __enter__(self):
        """
        Disables the gradients.

        Args:
            self: (todo): write your description
        """
        _set_grad_enabled(True)

    def __exit__(self, *args):
        """
        Exit the command to exit.

        Args:
            self: (todo): write your description
        """
        _set_grad_enabled(self.prev)
        return False


def is_grad_enabled():
    """Is auto-grad enabled?

    Returns
    -------
    bool
        **True** if enabling auto-grad.

    """
    return _GLOBAL_GRAD_OPTION.enabled


class no_grad(object):
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
        """
        Enter the gradients.

        Args:
            self: (todo): write your description
        """
        _set_grad_enabled(False)

    def __exit__(self, *args):
        """
        Exit the command to exit.

        Args:
            self: (todo): write your description
        """
        _set_grad_enabled(self.prev)
        return False


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
        """
        Enter the callable

        Args:
            self: (todo): write your description
        """
        pass

    def __exit__(self, *args):
        """
        Exit the command to exit.

        Args:
            self: (todo): write your description
        """
        _set_grad_enabled(self.prev)
        return False


def _set_grad_enabled(enabled=True):
    """Set the status of grad option."""
    global _GLOBAL_GRAD_OPTION
    _GLOBAL_GRAD_OPTION.enabled = enabled


_GLOBAL_GRAD_OPTION = tls.Constant(enabled=True)
