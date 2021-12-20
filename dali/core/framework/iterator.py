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

import ctypes

try:
    from nvidia.dali.backend import TensorCPU
    from nvidia.dali.backend import TensorGPU
except ImportError:
    TensorCPU = object
    TensorGPU = object

from dragon.core.device import cuda
from dragon.core.framework import device_spec
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor
from dragon.vm.dali.core.framework import types


class Iterator(object):
    """The base iterator wrapped for a pipeline."""

    def __init__(self, pipeline):
        """Create an ``Iterator``.

        Parameters
        ----------
        pipeline : dragon.vm.dali.Pipeline
            The pipeline to use.

        """
        self._pipe = pipeline
        self._workspace_id = id(workspace.get_workspace())

        # Build pipeline and cache the first batch.
        with self._api_scope():
            self._pipe.build()
            if self._pipe.device_id is not None:
                # Enforce the correct device of current process
                # to initialize cuda handles instead of device 0.
                cuda.set_device(self._pipe.device_id)
            self._pipe.schedule_run()
            self._copies = None
            self._first_batch = None
            self._first_batch = self.__next__()

    @property
    def batch_size(self):
        """Return the batch size of pipeline.

        Returns
        -------
        int
            The batch size.

        """
        return self._pipe.batch_size

    @property
    def handlers(self):
        """Define functions to process pipeline outputs.

        Returns
        -------
        Sequence[Tuple[Sequence[int], callable]]
            The handlers.

        """
        return [(None, self.copy_handler)]

    def copy_handler(self, tensors):
        """Handler to copy the data of tensors."""
        tensors = [t.as_tensor() for t in tensors]
        # Prepare the collection if it not created.
        if self._copies is None:
            self._copies = []
            for tensor in tensors:
                self._copies.append(self.new_tensor(
                    shape=tensor.shape(),
                    dtype=str(types.np_dtype(tensor.dtype())),
                    device=self.new_device(
                        device_type=('cuda' if isinstance(tensor, TensorGPU)
                                     else 'cpu'),
                        device_index=self._pipe.device_id)))
        # Transfer the data: DALI => Storage
        for i, tensor in enumerate(tensors):
            self._transfer_tensor(tensor, self._copies[i])
        return self._copies

    def epoch_size(self, reader='Reader'):
        """Return the epoch size of specified reader.

        Parameters
        ----------
        reader : str, optional, default='Reader'
            The reader name.

        Returns
        -------
        int
            The epoch size.

        """
        return self._pipe.epoch_size(reader)

    def get(self):
        """Return the next batch of data.

        Alias for ``self.__next__(...)``.

        Returns
        -------
        Sequence[dragon.Tensor]
            The output tensors.

        """
        return self.next()

    def next(self):
        """Return the next batch of data.

        Alias for ``self.__next__(...)``.

        Returns
        -------
        Sequence[dragon.Tensor]
            The output tensors.

        """
        return self.__next__()

    @staticmethod
    def new_device(device_type, device_index):
        """Return a new device abstraction."""
        return device_spec.DeviceSpec(device_type, device_index)

    @staticmethod
    def new_tensor(shape, dtype, device):
        """Return a new tensor abstraction."""
        return Tensor(shape=shape, dtype=dtype, device=device)

    def __iter__(self):
        """Return the iterator self."""
        return self

    def __next__(self):
        """Return the next batch of data."""
        # Return and reset the first batch if necessary.
        if self._first_batch is not None:
            outputs = self._first_batch
            self._first_batch = None
            return outputs

        # Verify the workspace is unchanged.
        # If not, drop all the relevant resources.
        self._check_workspace()

        with self._api_scope():
            # Block until the batch is completed.
            # Then, steal the reference of returns.
            pipe_returns = self._pipe.share_outputs()

        # Apply the handlers to process returns.
        # Outputs will be collected independently.
        outputs = []
        for indices, handler in self.handlers:
            if indices is None:
                indices = list(range(len(pipe_returns)))
            tensors = [pipe_returns[i] for i in indices]
            outputs.append(handler(tensors))

        # Release the scheduling mutex.
        with self._api_scope():
            self._pipe.release_outputs()
            self._pipe.schedule_run()

        # Return the outputs.
        return outputs[0] if len(outputs) == 1 else outputs

    def _api_scope(self):
        """Return the context-manger of pipeline api type."""
        return self._pipe._check_api_type_scope(types.PIPELINE_API_ITERATOR)

    def _check_workspace(self):
        """Monitor the workspace to reset invalid resources."""
        workspace_id = id(workspace.get_workspace())
        if workspace_id != self._workspace_id:
            self._copies = None
            self._workspace_id = workspace_id

    def _transfer_tensor(self, dali_tensor, target_tensor):
        """Transfer the dali tensor to the target."""
        target_shape = dali_tensor.shape()
        device = self.new_device(
            device_type='cuda' if isinstance(
                dali_tensor, TensorGPU) else 'cpu',
            device_index=self._pipe.device_id)
        if hasattr(target_tensor, '_device'):
            target_tensor._device = device
        impl = target_tensor._impl
        if target_shape != list(target_tensor.shape):
            new_capacity = not impl.Reshape(target_shape)
            if new_capacity:
                impl.mutable_data('cpu')
        if device.type == 'cuda':
            impl.ToCUDA(device.index)
            pointer = ctypes.c_void_p(impl.mutable_data('cuda'))
        else:
            pointer = ctypes.c_void_p(impl.mutable_data('cpu'))
        dali_tensor.copy_to_external(pointer)
