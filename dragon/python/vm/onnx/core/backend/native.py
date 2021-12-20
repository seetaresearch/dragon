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
    namedtupledict = collections.namedtuple

from dragon.core.autograph.graph_lib import GraphLib
from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor
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
        execute_ws = workspace.get_workspace()
        if device.type == DeviceType.CPU:
            device_type, device_index = 'cpu', 0
        elif device.type == DeviceType.CUDA:
            device_type, device_index = 'cuda', device.device_id
        else:
            raise ValueError('Unsupported device type: ' + device.type)
        with context.device(device_type, device_index):
            self._context = GraphLib.from_onnx(model)
        self._input_dict = collections.OrderedDict()
        self._output_dict = collections.OrderedDict()
        for input in self._context._def.input:
            impl = execute_ws.get_tensor(input)
            self._input_dict[input] = Tensor(impl=impl)
        for output in self._context._def.output:
            impl = execute_ws.get_tensor(output)
            self._output_dict[output] = Tensor(impl=impl)
        self._output_tuple = namedtupledict('Outputs', self._context._def.output)

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
        self._context.run()
        return self._output_tuple(*self._output_dict.values())


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
            ``True`` if device is supported otherwise ``False``.

        """
        device = Device(device_str)
        if device.type in (DeviceType.CPU, DeviceType.CUDA):
            return True
        return False


prepare = DragonBackend.prepare
run_model = DragonBackend.run_model
supports_device = DragonBackend.supports_device
