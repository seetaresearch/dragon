# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Implement some magic tricks to simplify the operator exporting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from dragon.core.tensor import Tensor


INT_MAX = 2147483647


class OpSchema(object):
    """Check the number of inputs, or convert to Tensors."""

    @classmethod
    def Inputs(cls, min_num, max_num=None):
        def Decorator(op_func):
            def Verify(inputs, min_num, max_num):
                if min_num == max_num and min_num == 0: return
                # It is a special case
                if min_num != max_num and min_num == 1:
                    if isinstance(inputs, Tensor):
                        inputs = [inputs]
                # Type checking
                if min_num == max_num and min_num == 1:
                    if not isinstance(inputs, Tensor):
                        raise ValueError(
                            'Excepted a Tensor as inputs, '
                                'got {}.'.format(type(inputs).__name__))
                else:
                    if not isinstance(inputs, list) or \
                            not all([isinstance(input, Tensor) for input in inputs]):
                        print(inputs)
                        raise ValueError('The inputs should be a list of Tensor.')
                # Range checking
                if not isinstance(inputs, list): inputs = [inputs]
                if len(inputs) < min_num or len(inputs) > max_num:
                    raise ValueError(
                        'The inputs size {}, is not in range: '
                            '[min={}, max={}].'.format(
                                len(inputs), min_num, max_num))
            @functools.wraps(op_func)
            def Impl(*args, **kwargs):
                if len(args) == 0:
                    raise ValueError('Excepted the first argument as inputs.')
                inputs = args[0]
                Verify(inputs, min_num, max_num if max_num else min_num)
                return op_func(*args, **kwargs)
            return Impl
        return Decorator

    @classmethod
    def ConvertConstantInputs(cls):
        def Decorator(op_func):
            @functools.wraps(op_func)
            def Impl(*args, **kwargs):
                inputs = args[0]
                if isinstance(inputs, (list, tuple)):
                    for idx, input in enumerate(inputs):
                        if not isinstance(input, Tensor):
                            inputs[idx] = Tensor.Convert(input, dtype=None)
                    return op_func(inputs + list(args[1:]), **kwargs)
                else:
                    if not isinstance(inputs, Tensor):
                        inputs = Tensor.Convert(inputs, dtype=None)
                    return op_func([inputs] + list(args[1:]), **kwargs)
            return Impl
        return Decorator


class ArgumentHelper(object):
    """Generate the descriptor for dynamic arguments."""

    @classmethod
    def Desc(cls, name, as_target=True):
        def Decorator(op_func):
            @functools.wraps(op_func)
            def Impl(*args, **kwargs):
                def Generator(arguments):
                    property = arguments.get(name, None)
                    if property is None: return arguments
                    if isinstance(property, Tensor):
                        if as_target:
                            if not 'extra_inputs' in arguments:
                                arguments['extra_inputs'] = []
                            arguments['extra_inputs'].extend([property])
                        arguments[name] = None
                        arguments[name + '_desc'] = property.name
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): Generator})
                return op_func(*args, **kwargs)
            return Impl
        return Decorator

    @classmethod
    def RepeatedDesc(cls, name, name_v2=None, dtype='int64', as_target=True):
        def Decorator(op_func):
            @functools.wraps(op_func)
            def Impl(*args, **kwargs):
                def Generator(arguments):
                    properties = arguments.get(name, None)
                    if properties is None: return arguments
                    desc_name = name_v2 if name_v2 else name
                    if name_v2: del arguments[name]
                    if not isinstance(properties, (list, tuple)):
                        properties = [properties]
                    # Check whether to use desc
                    tensor_in_properties = False
                    for property in properties:
                        if isinstance(property, Tensor):
                            tensor_in_properties = True
                    if tensor_in_properties:
                        properties_t = []
                        for property in properties:
                            if isinstance(property, Tensor):
                                if as_target:
                                    if not 'extra_inputs' in arguments:
                                        arguments['extra_inputs'] = []
                                    arguments['extra_inputs'].extend([property])
                                properties_t.append(property.name)
                            else:
                                properties_t.append(Tensor.Convert(property, dtype=dtype).name)
                        arguments[desc_name] = None
                        arguments[desc_name + '_desc'] = properties_t
                    else:
                        arguments[desc_name] = properties
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): Generator})
                return op_func(*args, **kwargs)
            return Impl
        return Decorator


def ParseArgs(locals):
    """Parse all the arguments into a dict."""
    __all__ = locals
    kwargs = __all__['kwargs']; del __all__['kwargs']
    desc_generators = {}
    for k, v in kwargs.items():
        if 'gen_desc' in k: desc_generators[k] = v
    for k in desc_generators.keys(): del kwargs[k]
    for v in desc_generators.values(): __all__ = v(__all__)
    return dict(__all__, **kwargs)