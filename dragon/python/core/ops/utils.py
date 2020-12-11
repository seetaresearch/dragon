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
"""Utilities for executing operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.framework import types
from dragon.core.autograph.tensor import Tensor
from dragon.core.util import decorator
from dragon.core.util import nest


class OpSchema(object):
    """Match arguments to the defined schema."""

    @classmethod
    def num_inputs(cls, min_num, max_num=None):
        def verify(inputs, min_num, max_num):
            if min_num == max_num and min_num == 0:
                return
            inputs = nest.flatten(inputs)
            if len(inputs) < min_num or len(inputs) > max_num:
                raise ValueError(
                    'The number of <inputs> is {}, '
                    'not in range: [min={}, max={}].'
                    .format(len(inputs), min_num, max_num))

        def decorated(inner_function):
            def wrapper(*args, **kwargs):
                if len(args) == 0:
                    raise ValueError('Excepted the first argument as <inputs>.')
                inputs = args[0]
                verify(inputs, min_num, max_num if max_num else min_num)
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated


class ArgHelper(object):
    """Generate and parse the descriptor for tensor arguments."""

    @staticmethod
    def desc(name, as_target=True):
        """Add desc for a single argument."""
        def decorated(inner_function):
            def wrapper(*args, **kwargs):
                def generator(arguments):
                    arg = arguments.get(name, None)
                    if arg is None:
                        return arguments
                    if types.is_tensor(arg):
                        ArgHelper._convert_to_desc(arguments, name, arg, as_target)
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): generator})
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated

    @staticmethod
    def parse(locals):
        """Parse all the arguments into a dict."""
        __all__ = locals
        kwargs = __all__['kwargs']
        del __all__['kwargs']
        desc_generators = {}
        for k, v in kwargs.items():
            if 'gen_desc' in k:
                desc_generators[k] = v
        for k in desc_generators.keys():
            kwargs.pop(k)
        for v in desc_generators.values():
            __all__ = v(__all__)
        return dict(__all__, **kwargs)

    @staticmethod
    def repeated_desc(name, name_v2=None, dtype='int64', as_target=True):
        """Add desc for a repeated argument."""
        def decorated(inner_function):
            def wrapper(*args, **kwargs):
                def generator(arguments):
                    arg = arguments.get(name, None)
                    if arg is None:
                        return arguments
                    key = name_v2 if name_v2 else name
                    if name_v2:
                        arguments.pop(name)
                    if types.is_tensor(arg):
                        ArgHelper._convert_to_desc(arguments, key, arg, as_target)
                    else:
                        if any([types.is_tensor(ele) for ele in arg]):
                            ArgHelper._convert_to_descs(arguments, dtype, key, arg, as_target)
                        else:
                            arguments[key] = arg
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): generator})
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated

    @staticmethod
    def _convert_to_desc(arguments, name, arg, as_target=False):
        """Convert the argument to a desc."""
        if context.executing_eagerly():
            arguments[name] = arg.get_value().tolist()
            return arguments
        if as_target:
            if 'extra_inputs' not in arguments:
                arguments['extra_inputs'] = []
            arguments['extra_inputs'] += [arg]
        arguments.pop(name)
        arguments[name + '_desc'] = arg.id
        return arguments

    @staticmethod
    def _convert_to_descs(arguments, dtype, name, arg, as_target=False):
        """Convert the argument to a sequence of descs."""
        if context.executing_eagerly():
            for i, ele in enumerate(arg):
                if types.is_tensor(ele):
                    arg[i] = ele.get_value().tolist()
            arguments[name] = arg
        else:
            descs = []
            for i, ele in enumerate(arg):
                if types.is_tensor(ele):
                    if as_target:
                        if 'extra_inputs' not in arguments:
                            arguments['extra_inputs'] = []
                        arguments['extra_inputs'] += [ele]
                    descs.append(ele.id)
                else:
                    descs.append(Tensor.from_value(ele, dtype, 'DescConst').id)
            arguments.pop(name)
            arguments[name + '_descs'] = descs
