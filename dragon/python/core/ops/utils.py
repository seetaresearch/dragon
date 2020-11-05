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
        """
        Decorator to add num_inputs ( min_num.

        Args:
            cls: (todo): write your description
            min_num: (int): write your description
            max_num: (int): write your description
        """
        def verify(inputs, min_num, max_num):
            """
            Verify that the inputs.

            Args:
                inputs: (array): write your description
                min_num: (int): write your description
                max_num: (int): write your description
            """
            if min_num == max_num and min_num == 0:
                return
            inputs = nest.flatten(inputs)
            if len(inputs) < min_num or len(inputs) > max_num:
                raise ValueError(
                    'The number of <inputs> is {}, '
                    'not in range: [min={}, max={}].'
                    .format(len(inputs), min_num, max_num)
                )

        def decorated(inner_function):
            """
            Decorator to add a function to the wrapped function.

            Args:
                inner_function: (todo): write your description
            """
            def wrapper(*args, **kwargs):
                """
                Decorator to callable.

                Args:
                """
                if len(args) == 0:
                    raise ValueError('Excepted the first argument as <inputs>.')
                inputs = args[0]
                verify(inputs, min_num, max_num if max_num else min_num)
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated


class ArgHelper(object):
    """Generate the descriptor for dynamic arguments."""

    @classmethod
    def desc(cls, name, as_target=True):
        """
        Decorator for a function.

        Args:
            cls: (todo): write your description
            name: (str): write your description
            as_target: (todo): write your description
        """
        def decorated(inner_function):
            """
            Decorator for decorators.

            Args:
                inner_function: (todo): write your description
            """
            def wrapper(*args, **kwargs):
                """
                Decorator for the given arguments.

                Args:
                """
                def generator(arguments):
                    """
                    Generate a generator for the given arguments.

                    Args:
                        arguments: (todo): write your description
                    """
                    arg = arguments.get(name, None)
                    if arg is None:
                        return arguments
                    if types.is_tensor(arg):
                        if context.executing_eagerly():
                            arguments[name] = arg.get_value().tolist()
                            return arguments
                        if as_target:
                            if 'extra_inputs' not in arguments:
                                arguments['extra_inputs'] = []
                            arguments['extra_inputs'].extend([arg])
                        arguments.pop(name)
                        arguments[name + '_desc'] = arg.id
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): generator})
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated

    @classmethod
    def repeated_desc(cls, name, name_v2=None, dtype='int64', as_target=True):
        """
        Decorator for the decorated function.

        Args:
            cls: (callable): write your description
            name: (str): write your description
            name_v2: (str): write your description
            dtype: (todo): write your description
            as_target: (todo): write your description
        """
        def decorated(inner_function):
            """
            Decorator to decorator.

            Args:
                inner_function: (todo): write your description
            """
            def wrapper(*args, **kwargs):
                """
                Create a function for tensor.

                Args:
                """
                def generator(arguments):
                    """
                    Generate a generator for a generator.

                    Args:
                        arguments: (todo): write your description
                    """
                    arg = arguments.get(name, None)
                    if arg is None:
                        return arguments
                    key = name_v2 if name_v2 else name
                    if name_v2:
                        arguments.pop(name)
                    if types.is_tensor(arg):
                        if context.executing_eagerly():
                            arguments[key] = arg.get_value().tolist()
                            return arguments
                        arguments.pop(key)
                        arguments[key + '_desc'] = arg.id
                        if as_target:
                            if 'extra_inputs' not in arguments:
                                arguments['extra_inputs'] = []
                            arguments['extra_inputs'] += [arg]
                    else:
                        has_tensor = False
                        arg = nest.flatten(arg)
                        for e in arg:
                            if types.is_tensor(e):
                                has_tensor = True
                                break
                        if has_tensor:
                            if context.executing_eagerly():
                                for i, e in enumerate(arg):
                                    if types.is_tensor(e):
                                        arg[i] = e.get_value().tolist()
                                arguments[key] = arg
                            else:
                                descs = []
                                for i, e in enumerate(arg):
                                    if types.is_tensor(e):
                                        if as_target:
                                            if 'extra_inputs' not in arguments:
                                                arguments['extra_inputs'] = []
                                            arguments['extra_inputs'] += [e]
                                        descs.append(e.id)
                                    else:
                                        descs.append(Tensor.convert_to(e, dtype).id)
                                arguments.pop(key)
                                arguments[key + '_descs'] = descs
                        else:
                            arguments[key] = arg
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): generator})
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated


def parse_args(locals):
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
