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
"""Native ONNX backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy

try:
    import onnx
    from onnx.backend.base import Backend
    from onnx.backend.base import BackendRep as ONNXBackendRep
    from onnx.backend.base import Device
    from onnx.backend.base import DeviceType
    from onnx.backend.base import namedtupledict
except ImportError:
    from dragon.core.util import deprecation
    onnx = deprecation.NotInstalled('onnx')
    Backend = object
    ONNXBackendRep = object
    Device = deprecation.NotInstalled('onnx')
    DeviceType = deprecation.NotInstalled('onnx')

from dragon.core.autograph import function_lib
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import context
from dragon.core.framework import device_spec
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import nest


class BackendRep(ONNXBackendRep):
    """ONNX-Dragon backend to execute repeatedly."""

    def __init__(self, model, device, **kwargs):
        """Create a ``BackendRep``.

        Parameters
        ----------
        model : str
            The path of onnx model file.
        device : onnx.Device
            The executing device.

        """
        if not isinstance(device, Device):
            device = Device(device)
        graph_str = workspace.get_workspace().PrepareONNXModel(model)
        graph_def = dragon_pb2.GraphDef()
        graph_def.ParseFromString(graph_str)
        if device.type == DeviceType.CPU:
            device_type, device_index = 'cpu', 0
        elif device.type == DeviceType.CUDA:
            device_type, device_index = 'cuda', device.device_id
        else:
            raise ValueError('Unsupported device type: ' + device.type)
        with context.device(device_type, device_index):
            self._function = function_lib.Function(name='ONNXGraph') \
                                         .import_from(graph_def)
        self._input_dict = collections.OrderedDict(
            [(impl.name, EagerTensor(impl=impl, device=device_spec.DeviceSpec(
                device_type, device_index))) for impl in self._function.inputs])
        self._output_dict = collections.OrderedDict(
            [(impl.name, EagerTensor(impl=impl, device=device_spec.DeviceSpec(
                device_type, device_index))) for impl in self._function.outputs])

    def run(self, inputs, **kwargs):
        """Run the model.

        Parameters
        ----------
        inputs : Union[Sequence, Dict]
            The input arrays.

        Returns
        -------
        namedtuple
            The model outputs.

        """
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, dict):
            for name, value in inputs.items():
                self._input_dict[name]._impl.FromNumpy(value)
        elif nest.is_sequence(inputs):
            for ref, value in zip(self._input_dict.values(), inputs):
                ref._impl.FromNumpy(value)
        else:
            raise ValueError('Excepted sequence or dict inputs.')
        self._function.callback(return_outputs=False)
        named_outputs = namedtupledict('Outputs', list(self._output_dict.keys()))
        return named_outputs(*(self._output_dict.values()))


class DragonBackend(Backend):
    """ONNX-Dragon backend."""

    @classmethod
    def prepare(cls, model, device='CPU:0', **kwargs):
        """Create a backend to execute repeatedly.

        Parameters
        ----------
        model : str
            The path of onnx model file.
        device : str, optional, default='CPU:0'
            The executing device.

        Returns
        -------
        dragon.onnx.BackendRep
            The backend.

        """
        if not os.path.exists(model):
            raise ValueError('Model({}) is not existed.'.format(model))
        return BackendRep(model, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device='CUDA:0', **kwargs):
        """Execute an onnx model once.

        Parameters
        ----------
        model : str
            The path of onnx model file.
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
    def supports_device(cls, device_str):
        """Query if the given device is supported.

        Parameters
        ----------
        device_str : str
            The device descriptor.

        Returns
        -------
        bool
            **True** if device is supported otherwise **False**.

        """
        device = Device(device_str)
        if device.type in (DeviceType.CPU, DeviceType.CUDA):
            return True
        return False


prepare = DragonBackend.prepare
run_model = DragonBackend.run_model
supports_device = DragonBackend.supports_device
