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
"""Function implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import types
import weakref

import numpy

from dragon.core.autograph import context as eager_context
from dragon.core.autograph.graph_impl import GraphLib
from dragon.core.autograph.graph_impl import GraphExecutionContext
from dragon.core.framework import device_spec
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
    unbound_method = original_function.python_function
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
        original_function.python_function,
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
    def from_function_and_signature(python_function, input_signature):
        """Create a FunctionSpec from function and signature."""
        is_method = inspect.ismethod(python_function)
        fullargspec = inspect.getfullargspec(python_function)
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
    """Map a python function to callable graphs."""

    def __init__(self, python_function, input_signature=None):
        self._python_function = python_function
        self._spec = FunctionSpec.from_function_and_signature(
            python_function, input_signature)
        self._graphs = dict()
        self._inputs = collections.defaultdict(list)
        self._outputs = collections.defaultdict(list)
        self._output_indices = collections.defaultdict(list)
        self._descriptor_cache = weakref.WeakKeyDictionary()

    @property
    def graphs(self):
        """Return the graphs."""
        return self._graphs.get(id(workspace.get_workspace()), None)

    @graphs.setter
    def graphs(self, value):
        """Set the graphs."""
        self._graphs[id(workspace.get_workspace())] = value

    @property
    def inputs(self):
        """Return the inputs."""
        return self._inputs.get(id(workspace.get_workspace()), None)

    @inputs.setter
    def inputs(self, value):
        """Set the inputs."""
        self._inputs[id(workspace.get_workspace())] = value

    @property
    def input_signature(self):
        """Return the input signature."""
        return self._spec.input_signature

    @property
    def outputs(self):
        """Return the outputs."""
        return self._outputs.get(id(workspace.get_workspace()), None)

    @outputs.setter
    def outputs(self, value):
        """Set the outputs."""
        self._outputs[id(workspace.get_workspace())] = value

    @property
    def python_function(self):
        """Return the python function."""
        return self._python_function

    def __call__(self, *args, **kwargs):
        """Call the compiled graphs."""
        if self.graphs is None:
            # Graphs are not created on the first call.
            # Compile from the python function.
            inputs = []
            input_signature = self.input_signature
            for i in range(self._spec.num_inputs):
                name, shape, dtype = 'Input:%d' % i, None, None
                if input_signature is not None:
                    if i >= len(input_signature):
                        raise ValueError(
                            'When <input_signature> is provided, '
                            'only define arguments covered by it.\n'
                            'Got %d signature(s) and %d argument(s).'
                            % (len(input_signature), self._spec.num_inputs))
                    shape = input_signature[i].shape
                    dtype = input_signature[i].dtype
                inputs.append(Tensor(shape, dtype, name, symbolic=True))
            _, kwargs = self._spec.separate_inputs(*args, **kwargs)
            with eager_context.graph_mode():
                returns = nest.flatten(self._python_function(*inputs, **kwargs))
            outputs, dummies, graphs = [], [], []
            for obj in returns:
                if isinstance(obj, Tensor):
                    outputs.append(obj)
                else:
                    dummies.append(obj)
            if len(outputs) > 0:
                graphs.append(GraphLib.from_outputs(outputs))
            for obj in dummies:
                if isinstance(obj, GraphExecutionContext):
                    graphs.append(obj)
            self.inputs = inputs
            self.outputs = returns
            self.graphs = graphs
        # In this case, we have compiled graphs.
        inputs, _ = self._spec.separate_inputs(*args, **kwargs)
        for input, value in zip(self.inputs, inputs):
            if hasattr(value, 'numpy'):
                value = value.numpy()
            else:
                value = numpy.array(value, copy=False)
            input._impl.FromNumpy(value)
        for graph in self.graphs:
            graph.run()
        outputs = []
        for output in self.outputs:
            if isinstance(output, Tensor):
                device = device_spec.DeviceSpec(*output._impl.device)
                outputs.append(Tensor(impl=output._impl, device=device))
            else:
                outputs.append(output)
        return outputs[0] if len(outputs) == 1 else outputs

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
