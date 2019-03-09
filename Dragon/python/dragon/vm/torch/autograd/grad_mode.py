# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/autograd/grad_mode.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = [
    'is_grad_enabled',
    'no_grad',
    'enable_grad',
    'set_grad_enabled',
]


grad_option = {'enable_grad': True}


def is_grad_enabled():
    """Is auto-grad enabled?

    Returns
    -------
    boolean
        ``True`` if enabling auto-grad.

    """
    global grad_option
    return grad_option['enable_grad']


def _set_grad_enabled(enabled=True):
    global grad_option
    grad_option['enable_grad'] = enabled



class no_grad(object):
    """Context-manager that disabled gradient calculation.

    """
    def __init__(self):
        self.prev = is_grad_enabled()

    def __enter__(self):
        _set_grad_enabled(False)

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)
        return False


class enable_grad(object):
    """Context-manager that enables gradient calculation.

    """
    def __init__(self):
        self.prev = is_grad_enabled()

    def __enter__(self):
        _set_grad_enabled(True)

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)
        return False


class set_grad_enabled(object):
    """Context-manager that sets gradient calculation to on or off.

    Parameters
    ----------
    mode : boolean
        ``True`` if enabling auto-grad.

    """
    def __init__(self, mode):
        self.prev = is_grad_enabled()
        _set_grad_enabled(mode)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        _set_grad_enabled(self.prev)
        return False