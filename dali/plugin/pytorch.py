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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import dali
from dragon.vm import torch


class DALIGenericIterator(dali.Iterator):
    """The general DALI iterator for ``vm.torch`` package."""

    def __init__(self, pipeline):
        """Create a ``DALIGenericIterator``."""
        super(DALIGenericIterator, self).__init__(pipeline)

    def get(self):
        """Return the next batch of data.

        Alias for ``self.__next__(...)``.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output tensors.

        """
        return self.next()

    def next(self):
        """Return the next batch of data.

        Alias for ``self.__next__(...)``.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output tensors.

        """
        return self.__next__()

    @staticmethod
    def new_device(device_type, device_index):
        """Return a new device abstraction."""
        return torch.device(device_type, device_index)

    @staticmethod
    def new_tensor(shape, dtype, device):
        """Return a new tensor abstraction."""
        return torch.Tensor(*shape, dtype=dtype, device=device)

    def __next__(self):
        """Return the next batch of data.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output tensors.

        """
        return super(DALIGenericIterator, self).__next__()
