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
"""Operator schema."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph import op_args
from dragon.core.autograph import op_spec
from dragon.core.framework import types
from dragon.core.util import decorator
from dragon.core.util import nest


class OpSchema(object):
    """Class to standardize the operator calls."""

    # Aliases
    register_args = op_args.register
    register_spec = op_spec.register
    get_args = op_args.get
    get_spec = op_spec.get

    @staticmethod
    def parse_args(locals):
        """Parse all the arguments into a dict."""
        __all__ = locals
        kwargs = __all__.pop('kwargs')
        desc_generators = {}
        for k, v in kwargs.items():
            if k.startswith('gen_desc_'):
                desc_generators[k] = v
        for k in desc_generators.keys():
            kwargs.pop(k)
        for v in desc_generators.values():
            __all__ = v(__all__)
        return {**__all__, **kwargs}

    @classmethod
    def num_inputs(cls, min_num, max_num=None):
        """Verify the number of inputs."""
        max_num = max_num if max_num else min_num

        def decorated(inner_function):
            def wrapper(*args, **kwargs):
                if len(args) == 0:
                    raise ValueError('Excepted the first argument is <inputs>.')
                inputs = nest.flatten(args[0])
                if len(inputs) < min_num or len(inputs) > max_num:
                    raise ValueError(
                        'The number of <inputs> is {}, '
                        'not in range: [min={}, max={}].'
                        .format(len(inputs), min_num, max_num))
                return inner_function(inputs, *args[1:], **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated

    @staticmethod
    def convert_arg(name, name_v2=None, as_target=True):
        """Convert argument to match the execution."""
        def decorated(inner_function):
            def wrapper(*args, **kwargs):
                def generator(arguments):
                    arg = arguments.get(name, None)
                    if arg is None:
                        return arguments
                    key = name_v2 if name_v2 else name
                    if name_v2 is not None:
                        arguments[key] = arguments.pop(name)
                    if types.is_tensor(arg):
                        OpSchema._convert_to_desc(
                            arguments, key, arg, as_target)
                    return arguments
                kwargs.update({'gen_desc_{}'.format(name): generator})
                return inner_function(*args, **kwargs)
            return decorator.make_decorator(inner_function, wrapper)
        return decorated

    @staticmethod
    def _convert_to_desc(arguments, name, arg, as_target=False):
        """Convert argument to a desc."""
        if context.executing_eagerly():
            arguments[name] = arg.numpy().tolist()
            return arguments
        if as_target:
            if 'extra_inputs' not in arguments:
                arguments['extra_inputs'] = []
            arguments['extra_inputs'] += [arg]
        if name in arguments:
            arguments.pop(name)
        arguments[name + '_desc'] = arg.id
        return arguments
