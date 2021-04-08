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
#     <https://github.com/onnx/onnx-tensorrt/blob/master/onnx_tensorrt/tensorrt_engine.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

try:
    from pycuda import driver
    from pycuda import gpuarray
    from pycuda import autoinit
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    from dragon.core.util import deprecation
    driver = deprecation.NotInstalled('pycuda')
    gpuarray = deprecation.NotInstalled('pycuda')
    autoinit = deprecation.NotInstalled('pycuda')
    trt = deprecation.NotInstalled('tensorrt')
    TRT_LOGGER = deprecation.NotInstalled('tensorrt')

from dragon.core.framework import device_spec
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor
from dragon.core.util import logging
from dragon.core.util import six


class Binding(object):
    """The binding wrapper for an input or output."""

    def __init__(self, cuda_engine, execution_context, idx_or_name, device_id):
        """Create a ``Binding``.

        Parameters
        ----------
        cuda_engine : tensorrt.ICudaEngine
            The built cuda engine.
        execution_context : tensorrt.IExecutionContext
            The execution context.
        idx_or_name : Union[int, str]
            The binding index or name.
        device_id : int, optional, default=0
            The index of executing device.

        """
        self._device_id = device_id
        self._context = execution_context

        if isinstance(idx_or_name, six.string_types):
            self._name = idx_or_name
            self._index = cuda_engine.get_binding_index(self._name)
            if self._index == -1:
                raise IndexError('Binding name not found: %s' % self._name)
        else:
            self._index = idx_or_name
            self._name = cuda_engine.get_binding_name(self._index)
            if self._name is None:
                raise IndexError('Binding index out of range: %i' % self._index)

        dtype_map = {
            trt.DataType.FLOAT: 'float32',
            trt.DataType.HALF: 'float16',
            trt.DataType.INT8: 'int8',
        }

        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = 'int32'

        self._is_input = cuda_engine.binding_is_input(self._index)
        self._dtype = dtype_map[cuda_engine.get_binding_dtype(self._index)]
        self._shape = tuple(cuda_engine.get_binding_shape(self.index))
        self._has_dynamic_shape = -1 in self._shape

        self._host_buf, self._device_buf = None, None
        self._host_tensor, self._device_tensor = None, None
        self._host_opt, self._device_opt = None, None

    @property
    def device_buffer(self):
        """Return the device buffer.

        Returns
        -------
        pycuda.gpuarray.GPUArray
            The pycuda array taking the data.

        """
        if self._device_buf is None:
            self._device_buf = gpuarray.empty(self._shape, self._dtype)
        return self._device_buf

    @property
    def device_dlpack(self):
        """Return the dlpack tensor wrapping device buffer.

        Returns
        -------
        PyCapsule
            The dlpack tensor object.

        """
        if self._device_tensor is None:
            spec = device_spec.DeviceSpec('cuda', self.device_id)
            self._device_opt = spec.to_proto(serialized=True)
            default_ws = workspace.get_workspace()
            impl = default_ws.create_tensor(scope='DLPack')
            impl.FromPointer(self._shape, self._dtype,
                             self._device_opt, self.device_buffer.ptr)
            self._device_tensor = Tensor(impl=impl, deleter=default_ws._handle_pool)
        return self._device_tensor._impl.ToDLPack(self._device_opt, True)

    @property
    def device_id(self):
        """Return the index of binding device.

        Returns
        -------
        int
            The device index.

        """
        return self._device_id

    @property
    def dtype(self):
        """Return binding data type.

        Returns
        -------
        str
            The binding data type.

        """
        return self._dtype

    @property
    def host_buffer(self):
        """Return the host buffer.

        Returns
        -------
        numpy.array
            The numpy array taking the data.

        """
        if self._host_buf is None:
            self._host_buf = driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def host_dlpack(self):
        """Return the dlpack tensor wrapping host buffer.

        Returns
        -------
        PyCapsule
            The dlpack tensor object.

        """
        if self._host_tensor is None:
            spec = device_spec.DeviceSpec('cpu')
            self._host_opt = spec.to_proto(serialized=True)
            default_ws = workspace.get_workspace()
            impl = default_ws.create_tensor(scope='DLPack')
            impl.FromPointer(self._shape, self._dtype,
                             self._host_opt, self.host_buffer.ctypes.data)
            self._host_tensor = Tensor(impl=impl, deleter=default_ws._handle_pool)
        return self._host_tensor._impl.ToDLPack(self._host_opt, True)

    @property
    def index(self):
        """Return the binding index.

        Returns
        -------
        int
            The binding index.

        """
        return self._index

    @property
    def is_input(self):
        """Whether this binding is an input.

        Returns
        -------
        bool
            ``True`` if binding is an input.

        """
        return self._is_input

    @property
    def name(self):
        """Return the binding name.

        Returns
        -------
        str
            The binding name.

        """
        return self._name

    @property
    def shape(self):
        """Return the binding shape.

        Returns
        -------
        Tuple[int]
            The binging shape.

        """
        return self._shape

    def get_async(self, stream):
        """Copy and return the host buffer data.

        Parameters
        ----------
        stream : pycuda.driver.Stream
            The cuda stream to copy data.

        Returns
        -------
        numpy.array
            The numpy array taking the data.

        """
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst

    def _check_size(self, new_shape):
        """Check whether the size is changed."""
        if self._shape != new_shape:
            if not (self._shape == (1,) and new_shape == ()):
                return True
        return False

    def _reset_buffer(self):
        """Reset both the host and device buffer."""
        self._host_buf, self._device_buf = None, None
        self._host_tensor, self._device_tensor = None, None
        self._host_opt, self._device_opt = None, None

    def _set_shape(self, new_shape=None):
        """Set a new shape and reset buffers if necessary."""
        if self._is_input:
            new_shape = tuple(new_shape)
            if self._check_size(new_shape):
                if self._has_dynamic_shape:
                    self._shape = new_shape
                    self._context.set_binding_shape(self._index, new_shape)
                    self._reset_buffer()
                else:
                    raise ValueError(
                        'Wrong shape for input "%s".\n'
                        'Expected %s, got %s.' %
                        (self._name, self._shape, new_shape)
                    )
        else:
            new_shape = tuple(self._context.get_binding_shape(self._index))
            if self._check_size(new_shape):
                self._shape = new_shape
                self._reset_buffer()


class Engine(object):
    """The executing engine with bindings."""

    def __init__(self, cuda_engine, device_id=0):
        """Create an ``Engine``.

        Parameters
        ----------
        cuda_engine : tensorrt.ICudaEngine
            The built cuda engine.
        device_id : int, optional, default=0
            The index of executing device.

        """
        # Create executing resources.
        self._cuda_engine = cuda_engine
        self._device_id = device_id
        self._context = cuda_engine.create_execution_context()
        self._stream = driver.Stream(0)

        # Create bindings.
        num_binding = self._cuda_engine.num_bindings
        self._bindings = [Binding(cuda_engine, self._context, i, device_id)
                          for i in range(num_binding)]
        self._inputs = [b for b in self._bindings if b.is_input]
        self._outputs = [b for b in self._bindings if not b.is_input]

        # Report the engine info.
        logging.info('TensorRT engine built.')
        binding_info = 'InputInfo: {\n'
        for b in self._inputs:
            binding_info += '  * Binding("{}", shape={}, dtype={})\n' \
                            .format(b.name, b.shape, b.dtype)
        logging.info(binding_info + '}')
        binding_info = 'OutputInfo: {\n'
        for b in self._outputs:
            binding_info += '  * Binding("{}", shape={}, dtype={})\n' \
                            .format(b.name, b.shape, b.dtype)
        logging.info(binding_info + '}')

    @property
    def cuda_engine(self):
        """Return the built cuda engine.

        Returns
        -------
        tensorrt.ICudaEngine
            The cuda engine.

        """
        return self._cuda_engine

    @property
    def inputs(self):
        """Return the input bindings.

        Returns
        -------
        Sequence[dragon.vm.tensorrt.Binding]
            The input bindings.

        """
        return self._inputs

    @property
    def outputs(self):
        """Return the output bindings.

        Returns
        -------
        Sequence[dragon.vm.tensorrt.Binding]
            The input bindings.

        """
        return self._outputs

    def get_results(self):
        """Return the engine executing results."""
        return [output.get_async(self._stream) for output in self._outputs]

    def run(self, inputs, optimization_profile=None):
        """Execute the engine and return the results.

        Parameters
        ----------
        inputs : Union[Sequence, Dict]
            The input numpy arrays.
        optimization_profile : int, optional
            The index of optimization profile to use.

        Returns
        -------
        Sequence[numpy.ndarray]
            The output arrays.

        """
        if len(inputs) < len(self.inputs):
            raise ValueError(
                'Not enough inputs. Expected %i, got %i.'
                % (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]

        # Copy data to input buffer.
        for i, (array, binding) in enumerate(zip(inputs, self.inputs)):
            array = check_input_validity(i, array, binding)
            binding.device_buffer.set_async(array, self._stream)

        # Prepare output buffer based on input buffer.
        for binding in self.outputs:
            binding._set_shape()

        # Setup the context before executing.
        if optimization_profile is not None:
            self._context.active_optimization_profile = optimization_profile

        # Dispatch the executions.
        binding_pointers = [b.device_buffer.ptr for b in self._bindings]
        self._context.execute_async_v2(binding_pointers, self._stream.handle)

        # Copy results from device.
        results = self.get_results()
        self._stream.synchronize()

        return results

    def __del__(self):
        if self._cuda_engine is not None:
            del self._cuda_engine


def check_input_validity(index, array, binding):
    """Check the input validity."""
    binding._set_shape(array.shape)
    if array.dtype != binding.dtype:
        if array.dtype == numpy.int64 and \
                binding.dtype == numpy.int32:
            casted_array = numpy.array(array, copy=True, dtype='int32')
            if numpy.equal(array, casted_array).all():
                array = casted_array
            else:
                raise TypeError(
                    'Wrong dtype for input %i.\n'
                    'Expected %s, got %s. Cannot safely cast.' %
                    (index, binding.dtype, array.dtype))
        else:
            raise TypeError(
                'Wrong dtype for input %i.\n'
                'Expected %s, got %s.' %
                (index, binding.dtype, array.dtype))
    return array
