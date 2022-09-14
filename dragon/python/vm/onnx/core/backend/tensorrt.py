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
"""TensorRT ONNX backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

try:
    import onnx
    from onnx.backend.base import Backend
    from onnx.backend.base import BackendRep as ONNXBackendRep
    from onnx.backend.base import Device
    from onnx.backend.base import DeviceType
except ImportError:
    from dragon.core.util import deprecation
    onnx = deprecation.NotInstalled('onnx')
    Backend = object
    ONNXBackendRep = object
    Device = deprecation.NotInstalled('onnx')
    DeviceType = deprecation.NotInstalled('onnx')

from dragon.core.device import cuda
from dragon.core.util import six
from dragon.vm.onnx.core import helper as onnx_helper
from dragon.vm.tensorrt.core import engine
from dragon.vm.tensorrt.core.engine import trt
from dragon.vm.tensorrt.core.engine import TRT_LOGGER


class BackendRep(ONNXBackendRep):
    """ONNX-TensorRT backend to execute repeatedly."""

    def __init__(
        self,
        model,
        device,
        max_batch_size=32,
        max_workspace_size=None,
        optimization_profiles=None,
        serialize_engine=False,
    ):
        """Create a ``BackendRep``.

        Parameters
        ----------
        model : onnx.ModelProto
            The onnx model.
        device : onnx.Device
            The executing device.
        max_batch_size : int, optional, default=32
            The max batch size.
        max_workspace_size : int, optional
            The max workspace size in bytes.
        optimization_profiles : List[Dict], optional
            The optimization profiles.
        serialize_engine : bool, optional, default=False
            Whether to serialize engine into a file.

        """
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self._logger = TRT_LOGGER
        self._builder = trt.Builder(self._logger)
        self._builder_config = self._builder.create_builder_config()
        self._network = self._builder.create_network(
            flags=1 << (int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))
        self._parser = trt.OnnxParser(self._network, self._logger)

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ''):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        if not self._parser.parse(model_str):
            error = self._parser.get_error(0)
            msg = "While parsing node #%i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)

        if max_workspace_size is None:
            max_workspace_size = 1 << 28

        # Setup the builder.
        self._builder.max_batch_size = max_batch_size
        self._builder.max_workspace_size = max_workspace_size
        self._add_optimization_profiles(optimization_profiles)

        # Build and wrap for the cuda engine.
        if optimization_profiles is None:
            cuda_engine = self._builder.build_cuda_engine(self._network)
        else:
            cuda_engine = self._builder.build_engine(self._network, self._builder_config)
        if cuda_engine is None:
            raise RuntimeError("Failed to build TensorRT engine from network.")
        if serialize_engine:
            cuda_engine = self._serialize_deserialize(cuda_engine)
        self._engine = engine.Engine(cuda_engine, device.device_id)

        self._output_shapes = {}
        self._output_dtypes = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtypes[output.name] = output.type.tensor_type.elem_type

    @property
    def engine(self):
        """Return the executing engine.

        Returns
        -------
        dragon.vm.tensorrt.Engine
            The executing engine

        """
        return self._engine

    def run(self, inputs, optimization_profile=None, **kwargs):
        """Run the model.

        Parameters
        ----------
        inputs : Union[Sequence, Dict]
            The input arrays.
        optimization_profile : int, optional
            The index of optimization profile to use.

        Returns
        -------
        namedtuple
            The model outputs.

        """
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        outputs = self._engine.run(
            inputs, optimization_profile=optimization_profile)
        output_names = [output.name for output in self._engine.outputs]
        for i, (name, array) in enumerate(zip(output_names, outputs)):
            if self._output_dtypes[name] == onnx.TensorProto.INT64 and \
                    array.dtype == numpy.int32:
                outputs[i] = numpy.array(outputs[i], dtype=numpy.int64)
        return onnx_helper.namedtupledict('Outputs', output_names)(*outputs)

    def _add_optimization_profiles(self, profiles):
        """Add optimization profiles into builder config."""
        if profiles is None:
            return
        for profile in profiles:
            for input_name, selectors in profile.items():
                min_shape, opt_shape, max_shape = selectors
                if min_shape is None:
                    raise ValueError('Excepted the min shape for a valid profile.')
                opt_shape = min_shape if opt_shape is None else opt_shape
                max_shape = min_shape if max_shape is None else max_shape
                profile = self._builder.create_optimization_profile()
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                self._builder_config.add_optimization_profile(profile)

    def _serialize_deserialize(self, cuda_engine):
        runtime = trt.Runtime(TRT_LOGGER)
        serialized_engine = cuda_engine.serialize()
        del self._parser
        cuda_engine = runtime.deserialize_cuda_engine(serialized_engine)
        return cuda_engine

    def _set_device(self, device):
        self.device = device
        assert device.type == DeviceType.CUDA
        cuda.set_device(device.device_id)


class TensorRTBackend(Backend):
    """ONNX-TensorRT backend."""

    @classmethod
    def prepare(cls, model, device='CUDA:0', **kwargs):
        """Create a backend to execute repeatedly.

        Parameters
        ----------
        model : onnx.ModelProto
            The onnx model.
        device : str, optional
            The executing device.

        Returns
        -------
        tensorrt.onnx.BackendRep
            The backend.

        """
        return BackendRep(model, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device='CUDA:0', **kwargs):
        """Execute an onnx model once.

        Parameters
        ----------
        model : onnx.ModelProto
            The onnx model.
        inputs : Union[Sequence, Dict]
            The input arrays.
        device : str, optional
            The executing device.

        Returns
        -------
        namedtuple
            The model outputs.

        """
        return cls.prepare(model, device, **kwargs).run(inputs)

    @classmethod
    def run_node(cls, node, inputs, device='CUDA:0', **kwargs):
        """Execute an onnx node once.

        Parameters
        ----------
        node : onnx.NodeProto
            The onnx node.
        inputs : Union[Sequence, Dict]
            The input arrays.
        device : str, optional, default='CUDA:0'
            The executing device.

        Returns
        -------
        namedtuple
            The model outputs.

        """
        super(TensorRTBackend, cls).run_node(node, inputs, device)
        model = onnx_helper.make_model_from_node(node, inputs, use_weights=True)
        try:
            results = cls.prepare(model, device).run(inputs[:1])
        except RuntimeError:
            model = onnx_helper.make_model_from_node(node, inputs, use_weights=False)
            results = cls.prepare(model, device).run(inputs)
        return results

    @classmethod
    def supports_device(cls, device_str):
        """Query if the given device is supported.

        Parameters
        ----------
        device_str : str
            The device descriptor.

        Returns
        -------
        bool
            ``True`` if device is supported otherwise ``False``.

        """
        device = Device(device_str)
        return device.type == DeviceType.CUDA


prepare = TensorRTBackend.prepare
run_node = TensorRTBackend.run_node
run_model = TensorRTBackend.run_model
supports_device = TensorRTBackend.supports_device
