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

import numpy

from dragon.core.framework import config
from dragon.core.framework import proto_util
from dragon.vm.torch.core.autograd import execute


class Function(object):
    """Dispatch the tensor operation.

    Each tensor operation takes a function instance to dispatch the specific execution.

    To define a new function class, overrides ``attributes`` and ``forward``:

    ```python
    class MyReLU(autograd.Function):

        def __init__(self, key, dev, **kwargs):
            super(MyReLU, self).__init__(key, dev, **kwargs)
            self.alpha = kwargs.get('alpha', 0)

        def attributes(self):
            return {'op_type': 'Relu', 'arguments': {'alpha': float(self.alpha)}}

        def forward(self, input):
            outputs = [self.alloc()]
            return self.dispatch([input], outputs)
    ```

    Function is executed by instantiating attributes and applying to inputs:

    ```python
    def my_relu(input, alpha=0):
        return MyReLU.instantiate(input.device, alpha=alpha).apply(input)
    ```

    """

    def __init__(self, key, dev, **kwargs):
        """Create a ``Function``.

        Parameters
        ----------
        key : str
            The cache key.
        device : dragon.vm.torch.device
            The device spec.

        """
        super(Function, self).__init__()
        self._def = None
        self._cache_key = key
        self._device = dev
        self._arg_device = proto_util.get_device_option('cpu')
        self._arg_device = self._arg_device.SerializeToString()
        self._seed = kwargs.get('seed', config.config().random_seed)

    def alloc(self, out=None):
        """Return or bind the executing device to output tensor.

        Parameters
        ----------
        out : dragon.vm.torch.Tensor, optional
            The optional output tensor.

        Returns
        -------
        Union[dragon.vm.torch.device, dragon.vm.torch.Tensor]
            The executing device or output tensor.

        """
        if out is not None:
            out._device = self._device.copy()
            return out
        return self._device.copy()

    def apply(self, *args, **kwargs):
        """Apply this function to inputs."""
        return self.__call__(*args, **kwargs)

    def attributes(self):
        """Return the function attributes.

        Returns
        -------
        dict
            The attribute dict.

        """

    def dispatch(
        self,
        inputs,
        outputs,
        no_grad=False,
        check_device=True,
        callback=None,
    ):
        """Dispatch the execution."""
        if self._def is None:
            self._gen_def()
        if len(inputs) > 1 and check_device:
            self._check_device(inputs)
        return execute.run_operator(
            op_def=self._def,
            inputs=inputs,
            outputs=outputs,
            no_grad=no_grad,
            pre_callback=callback,
        )

    def feed_arg(self, ws, name, value, dtype='int64'):
        """Set the value of tensor argument."""
        ws.FeedTensor(name, numpy.array(value, dtype), self._arg_device)

    @classmethod
    def instantiate(cls, device, **kwargs):
        """Return an instance of this function.

        Parameters
        ----------
        device : dragon.vm.torch.device
            The executing device.

        Returns
        -------
        dragon.vm.torch.autograd.Function
            The function instance.

        """

        def keygen(cls, device, **kwargs):
            """Generate the cache key from device and attributes."""
            key = '%s/%s' % (cls.__name__, device)
            for v in kwargs.values():
                key += '/' + str(v)
            return key

        def creator(cls, cache_key, device, **kwargs):
            """Create and then cache a function."""
            function = cls(cache_key, device, **kwargs)
            _GLOBAL_CACHED_FUNCTIONS[cache_key] = function
            return function

        cache_key = keygen(cls, device, **kwargs)
        try:
            return _GLOBAL_CACHED_FUNCTIONS[cache_key]
        except KeyError:
            return creator(cls, cache_key, device, **kwargs)

    def forward(self, *inputs, **kwargs):
        """Define the execution."""

    def _gen_def(self):
        """Generate the OpDef from attributes."""
        attributes = self.attributes()
        self._def = proto_util.make_operator_def_cpp(
            name=attributes.get('name', 'Op'),
            cache_key=self._cache_key,
            op_type=attributes['op_type'],
            device_option=proto_util.get_device_option(
                self._device.type,
                self._device.index,
                self._seed,
            ),
            **attributes['arguments']
        )

    def _check_device(self, inputs):
        """Check the device of inputs."""
        for i, input in enumerate(inputs):
            if input._device != self._device:
                raise RuntimeError(
                    'Mismatched device between function and '
                    'element {} of input tensors. ({} vs. {})'
                    .format(i, self._device, input._device))

    def __call__(self, *args, **kwargs):
        """Call the ``self.forward(...)``."""
        return self.forward(*args, **kwargs)


# Define a global dict to cache the functions.
# This behavior is guaranteed to be thread-safe.
_GLOBAL_CACHED_FUNCTIONS = {}
