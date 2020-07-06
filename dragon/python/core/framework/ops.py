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

"""Utilities to fly an operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.op_def import OpDef
from dragon.core.eager import executor
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import config
from dragon.core.framework import proto_util
from dragon.core.framework import context
from dragon.core.framework import types
from dragon.core.framework import workspace


class Operator(object):
    """Wrapper to unify the symbolic and eager operator abstraction."""

    def __init__(self, key, dev, **kwargs):
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

    @classmethod
    def blend(cls, op_type=None, **kwargs):
        """Attach the OpDef to outputs."""
        op_type = op_type if op_type else cls.__name__
        return OpDef.apply(op_type, **kwargs)

    def dispatch(self, inputs, outputs, no_grad=False, callback=None):
        """Dispatch the execution."""
        if self._def is None:
            self._gen_def()
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
    def instantiate(cls, **kwargs):
        """Return a cached operator instance."""

        def keygen(cls, device, **kwargs):
            """Generate a cache key from device and attributes."""
            key = '%s/%s' % (cls.__name__, device)
            for v in kwargs.values():
                key += '/' + str(v)
            return key

        def creator(cls, cache_key, device, **kwargs):
            """Create and then cache a operator."""
            op = cls(cache_key, device, **kwargs)
            _GLOBAL_CACHED_OPERATORS[cache_key] = op
            return op

        device = context.get_device_spec()
        cache_key = keygen(cls, device, **kwargs)
        try:
            return _GLOBAL_CACHED_OPERATORS[cache_key]
        except KeyError:
            return creator(cls, cache_key, device, **kwargs)

    def forward(self, *inputs, **kwargs):
        """Define the execution."""
        pass

    def _gen_def(self):
        """Generate the OpDef from attributes."""
        attributes = self.attributes()
        self._def = proto_util.make_operator_cdef(
            name=attributes.get('name', 'GenericOp'),
            cache_key=self._cache_key,
            op_type=attributes['op_type'],
            device_option=proto_util.get_device_option(
                self._device.type,
                self._device.index,
                self._seed,
            ),
            **attributes['arguments']
        )

    def __call__(self, *args, **kwargs):
        """Call the ``self.forward(...)``."""
        return self.forward(*args, **kwargs)


def new_leaf(shape, dtype, device, trainable=False):
    """Return a leaf resource."""
    return EagerTensor(**locals())


def remove_binary_scalar(inputs):
    """Remove the scalar for binary ops."""
    if types.is_tensor(inputs[0]):
        # (Tensor, Number)
        inputs[1] = scalar_to_tensor(
            inputs[1],
            inputs[0].dtype,
        )
    else:
        # (Number, Tensor)
        inputs[0] = scalar_to_tensor(
            inputs[0],
            inputs[1].dtype,
        )
    return inputs


def scalar_to_tensor(input, dtype):
    """Return a cached scalar tensor."""
    if types.is_tensor(input):
        return input
    try:
        input = float(input)
    except (TypeError, ValueError):
        raise ValueError(
            '<input> should be a python number, got {}.'
            .format(type(input).__name__)
        )
    tid = '/share/scalar/{}/{}'.format(dtype, str(input))
    if not workspace.has_tensor(tid):
        workspace.feed_tensor(tid, numpy.array(input, dtype))
    return EagerTensor(
        id=tid,
        dtype=dtype,
        own_storage=False,
        requires_grad=False,
    )


# Define a global dict to cache the operators.
# This behavior is guaranteed to be thread-safe.
_GLOBAL_CACHED_OPERATORS = {}
