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
"""Basic prototypes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TensorBase(object):
    """Tensor base class."""


def is_tensor(blob):
    """Whether the given blob is a tensor object."""
    return isinstance(blob, TensorBase)
