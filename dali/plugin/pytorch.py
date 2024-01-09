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
