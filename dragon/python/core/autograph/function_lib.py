# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/def_function.py>
#
# ------------------------------------------------------------
"""Function library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import types
import weakref

import numpy

from dragon.core.autograph import context as eager_context
from dragon.core.autograph.graph_lib import GraphLib
from dragon.core.autograph.graph_lib import GraphExec
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor
from dragon.core.util import decorator
from dragon.core.util import inspect
from dragon.core.util import nest
from dragon.core.util import six


class MethodTarget(object):
    """Class to wrap the target into a instance method."""

    def __init__(self, target, original_function):
        self.weakrefself_target__ = target
        self.weakrefself_func__ = weakref.ref(original_function)

    @property
    def target(self):
        return self.weakrefself_target__()

    def call(self, args, kwargs):
        wrapped_fn = self.weakrefself_func__()
        if inspect.ismethod(wrapped_fn):
            wrapped_fn = six.get_unbound_function(wrapped_fn)
        return wrapped_fn(self.weakrefself_target__(), *args, **kwargs)


def class_method_to_instance_method(original_function, instance):
    """Patch a class method to an instance method.

    This function is required when an bound target has
    turned to unbound due to the unpack of decorator.

    """
    weak_instance = weakref.ref(instance)
    unbound_method = original_function.run_function
    target = MethodTarget(weak_instance, unbound_method)
    bound_method = types.MethodType(unbound_method, target)
    weak_bound_method_wrapper = None

    def bound_method_wrapper(*args, **kwargs):
        strong_bound_method_wrapper = weak_bound_method_wrapper()
        wrapped_fn = strong_bound_method_wrapper.__wrapped__
        if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
            wrapped_fn = unbound_method
            if inspect.ismethod(wrapped_fn):
                wrapped_fn = six.get_unbound_function(wrapped_fn)
            return wrapped_fn(weak_instance(), *args, **kwargs)
        return wrapped_fn(*args, **kwargs)

    weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)

    return decorator.make_decorator(
        original_function.run_function,
        type(original_function)(
            decorator.make_decorator(bound_method, bound_method_wrapper),
            input_signature=original_function.input_signature),
    )


class FunctionSpec(object):
    """Function spec."""

    def __init__(self, fullargspec, is_method, input_signature):
        self._fullargspec = fullargspec
        self._is_method = is_method
        self._input_signature = input_signature
        if self._is_method:
            args = fullargspec.args[1:]
        else:
            args = fullargspec.args
        self._default_values = fullargspec.defaults
        self._args_to_indices = {arg: i for i, arg in enumerate(args)}
        offset = len(args) - len(self._default_values or [])
        self._arg_indices_to_default_values = {
            offset + i: v for i, v in enumerate(self._default_values or [])
        }

    @property
    def input_signature(self):
        """Return the input signature."""
        return self._input_signature

    @property
    def num_inputs(self):
        """Return the number of defined inputs."""
        bound_num = len(self._fullargspec.args)
        return bound_num - 1 if self._is_method else bound_num

    @staticmethod
    def from_function_and_signature(run_function, input_signature):
        """Create a FunctionSpec from function and signature."""
        is_method = inspect.ismethod(run_function)
        fullargspec = inspect.getfullargspec(run_function)
        return FunctionSpec(fullargspec, is_method, input_signature)

    def separate_inputs(self, *args, **kwargs):
        # Check the match between arguments and signatures.
        if self._input_signature is not None:
            # In this case, extra kwargs are forbidden.
            if len(args) > len(self._input_signature):
                raise ValueError(
                    'When <input_signature> is provided, '
                    'only pass arguments covered by it.\n'
                    'Received %d argument(s).' % len(args))
        # Determine the args from kwargs and default-values.
        if not kwargs:
            # The simplest case: args only.
            inputs = args
            for i in sorted(self._arg_indices_to_default_values.keys()):
                if i >= len(args):
                    inputs += (self._arg_indices_to_default_values[i],)
        else:
            # Select from kwargs if defined in the spec.
            arg_indices_to_values = {
                i: v for i, v in self._arg_indices_to_default_values.items()
                if i >= len(args)
            }
            extra_args = {}
            for arg, value in kwargs.items():
                index = self._args_to_indices.get(arg, None)
                if index is not None:
                    arg_indices_to_values[index] = value
                else:
                    extra_args[arg] = value
            args2 = tuple(arg_indices_to_values[key]
                          for key in sorted(arg_indices_to_values))
            inputs, kwargs = args + args2, extra_args
        # Return the final inputs and kwargs.
        return inputs, kwargs


class FunctionGuard(object):
    """Map a run function to callable graphs."""

    def __init__(self, run_function, input_signature=None):
        self._run_function = run_function
        self._spec = FunctionSpec.from_function_and_signature(
            run_function, input_signature)
        self._attribute_cache = collections.defaultdict(dict)
        self._descriptor_cache = weakref.WeakKeyDictionary()

    @property
    def input_signature(self):
        """Return the input signature."""
        return self._spec.input_signature

    @property
    def run_function(self):
        """Return the run function."""
        return self._run_function

    def _build_graphs(self, *args, **kwargs):
        attributes = self._attribute_cache[workspace.get_workspace()]
        input_signature = self._spec.input_signature
        args, kwargs = self._spec.separate_inputs(*args, **kwargs)
        inputs = []
        for i in range(self._spec.num_inputs):
            input_spec = None
            if input_signature is not None:
                input_spec = input_signature[i]
            if not isinstance(args[i], Tensor) and input_spec is None:
                inputs.append(args[i])
                continue
            name = 'Input_%d' % (i + 1)
            shape = getattr(args[i], 'shape', None)
            dtype = getattr(args[i], 'dtype', None)
            if input_spec is not None:
                shape, dtype = input_spec.shape, input_spec.dtype
            inputs.append(Tensor(shape, dtype, name=name, symbolic=True))
        with eager_context.graph_mode():
            outputs = self._run_function(*inputs, **kwargs)
        graph_outputs, dummies, graphs = [], [], []
        for output in nest.flatten(outputs):
            if isinstance(output, Tensor):
                graph_outputs.append(output)
            else:
                dummies.append(output)
        if len(graph_outputs) > 0:
            graphs.append(GraphLib.from_outputs(graph_outputs))
        for obj in dummies:
            if isinstance(obj, GraphExec):
                graphs.append(obj)
        attributes['inputs'] = inputs
        attributes['outputs'] = outputs
        attributes['graphs'] = graphs
        return graphs

    def __call__(self, *args, **kwargs):
        """Call the built graphs."""
        execute_ws = workspace.get_workspace()
        attributes = self._attribute_cache[execute_ws]
        graphs = attributes.get('graphs', None)
        if graphs is None:
            graphs = self._build_graphs(*args, **kwargs)
        inputs = attributes['inputs']
        values, _ = self._spec.separate_inputs(*args, **kwargs)
        for input, value in zip(inputs, values):
            if not isinstance(input, Tensor):
                continue
            if hasattr(value, 'numpy'):
                value = value.numpy()
            else:
                value = numpy.array(value, copy=False)
            input._impl.FromNumpy(value)
        for graph in graphs:
            graph.run()
        return attributes['outputs']

    def __get__(self, instance, owner):
        """Override to patch the instance methods."""
        del owner
        if instance not in self._descriptor_cache:
            if instance is None:
                return self
            self._descriptor_cache[instance] = \
                class_method_to_instance_method(self, instance)
        return self._descriptor_cache[instance]


def function(func=None, input_signature=None):
    """Compile a function into the callable graph.

    Only the tensor operations could be compiled:

    ```python
    def foo(x, y):
        return dragon.math.add([x + y, x])

    bar = dragon.function(foo)
    print(bar(1, 2))
    ```

    Above usages which can simplified as follows:

    ```python
    @dragon.function
    def foo(x, y):
        return dragon.math.add([x + y, x])

    print(foo(1, 2))
    ```

    Tensor shape and dtype will be required sometimes:

    ```python
    @dragon.function(input_signature=[
        dragon.Tensor(shape=[], dtype='float32'),
        dragon.Tensor(shape=[], dtype='float32')])
    def foo(x, y):
        return dragon.math.add([x + y, x])

    print(foo(1, 2))
    ```

    Parameters
    ----------
    func : callable, optional
        The function to be compiled.
    input_signature : Sequence[dragon.Tensor], optional
        The tensors to hint the input.

    Returns
    -------
    callable
        A callable to execute the compiled function.

    """
    def decorated(inner_function):
        return decorator.make_decorator(
            inner_function,
            FunctionGuard(inner_function, input_signature))
    if func is not None:
        return decorated(func)
    return decorated
