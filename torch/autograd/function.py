# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import config
from dragon.core.framework import proto_util
from dragon.vm.torch import executor


class Function(object):
    def __init__(self, key, dev, **kwargs):
        super(Function, self).__init__()
        self._def = None
        self._cache_key = key
        self._device = dev
        self._arg_device = proto_util.get_device_option('cpu')
        self._arg_device = self._arg_device.SerializeToString()
        self._seed = kwargs.get('seed', config.config().random_seed)

    def alloc(self):
        """Return a device spec."""
        return self._device.copy()

    def apply(self, *args, **kwargs):
        """An alias of ``self.__call__(...)``"""
        return self.__call__(*args, **kwargs)

    def attributes(self):
        """Define the attributes to generate OpDef."""
        return {}

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
        if check_device:
            self._check_device(inputs)
        return executor.run_operator(
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
        """Return an instance of this function."""

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
        raise RuntimeError('The base function can not be called.')

    def _gen_def(self):
        """Generate the OpDef from attributes."""
        attributes = self.attributes()
        self._def = proto_util.make_operator_cdef(
            name='Generic',
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
        for ix, t in enumerate(inputs):
            if t._device != self._device:
                raise ValueError(
                    'Function is defined at {}, '
                    '\nFound Input({}) is at {}.'
                    .format(self._device, ix, t._device),
                )

    def __call__(self, *args, **kwargs):
        """Call the ``self.forward(...)``."""
        return self.forward(*args, **kwargs)


# Define a global dict to cache the functions.
# This behavior is guaranteed to be thread-safe.
_GLOBAL_CACHED_FUNCTIONS = {}
