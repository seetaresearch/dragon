# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings


class Variable(object):
    """The variable class."""

    def __new__(cls, tensor, requires_grad=False, volatile=False):
        if volatile:
            warnings.warn("volatile was removed and now has no effect. "
                          "Use `with torch.no_grad():` instead.", stacklevel=2)
        if requires_grad and volatile:
            raise RuntimeError("Variable can't be volatile and require_grad at the same time.")
        tensor.requires_grad = requires_grad
        return tensor
