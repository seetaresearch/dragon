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

"""Define the basic prototypes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# The tensor subclasses:
#  * Tensor (i.e., SymbolicTensor)
#  * EagerTensor
class TensorMetaclass(object):
    pass


def is_eager_tensor(blob):
    """Whether the given blob is an eager tensor."""
    return is_tensor(blob) and hasattr(blob, '__del__')


def is_symbolic_tensor(blob):
    """Whether the given blob is a symbolic tensor."""
    return is_tensor(blob) and not hasattr(blob, '__del__')


def is_tensor(blob):
    """Whether the given blob is a generic tensor."""
    return isinstance(blob, TensorMetaclass)
