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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_shape.py>
#
# ------------------------------------------------------------
"""Tensor shape utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TensorShape(tuple):
    """Represent the a sequence of dimensions."""

    def __init__(self, dims):
        """Create a ``TensorShape``.

        Parameters
        ----------
        dims : Sequence[int]
            The dimensions.

        """
        super(TensorShape, self).__init__()

    @property
    def dims(self):
        """Return the list of dimensions.

        Returns
        -------
        List[int]
            The dimensions.

        """
        return list(self)

    @property
    def ndims(self):
        """Return the number of dimensions.

        Deprecated. See ``TensorShape.rank``.

        Returns
        -------
        int
            The number of dimensions.

        """
        return len(self)

    @property
    def rank(self):
        """Return the rank of shape.

        Returns
        -------
        int
            The rank.

        """
        return len(self)

    def as_list(self):
        """Return the list of dimensions.

        Returns
        -------
        List[int]
            The dimensions.

        """
        return list(self)

    def __repr__(self):
        return "TensorShape({})".format(list(self))

    def __str__(self):
        if self.ndims == 1:
            return "(%s,)" % self.dims[0]
        else:
            return "(%s)" % ", ".join(str(d) for d in self.dims)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return TensorShape(self.dims[key])
        else:
            return self.dims[key]
