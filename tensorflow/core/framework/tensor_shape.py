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
"""Tensor shape utilities."""


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
