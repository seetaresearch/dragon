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

import collections
import weakref

from dragon.core.autograph import tape
from dragon.core.autograph.function_impl import FunctionSpec
from dragon.core.autograph.function_impl import class_method_to_instance_method
from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.util import decorator
from dragon.core.util import nest
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.tensor import Tensor


class FunctionGuard(object):
    """Map the python function to workspace-local functions."""

    def __init__(self, python_function, input_signature=None):
        self._python_function = python_function
        self._spec = FunctionSpec.from_function_and_signature(
            python_function, input_signature)
        self._defs = collections.defaultdict(list)
        self._inputs = collections.defaultdict(list)
        self._outputs = collections.defaultdict(list)
        self._descriptor_cache = weakref.WeakKeyDictionary()

    @property
    def defs(self):
        """Return the recorded operator defs."""
        return self._defs.get(id(workspace.get_workspace()), None)

    @defs.setter
    def defs(self, value):
        """Set the recorded operator defs."""
        self._defs[id(workspace.get_workspace())] = value

    @property
    def inputs(self):
        """Return the input symbols."""
        return self._inputs.get(id(workspace.get_workspace()), None)

    @inputs.setter
    def inputs(self, value):
        """Set the input symbols."""
        self._inputs[id(workspace.get_workspace())] = value

    @property
    def input_signature(self):
        """Return the defined input signature."""
        return self._spec.input_signature

    @property
    def outputs(self):
        """Return the output symbols."""
        return self._outputs.get(id(workspace.get_workspace()), None)

    @outputs.setter
    def outputs(self, value):
        """Set the output symbols."""
        self._outputs[id(workspace.get_workspace())] = value

    @property
    def python_function(self):
        """Return the defined python function."""
        return self._python_function

    def separate_inputs(self, *args, **kwargs):
        """Separate inputs from the call arguments."""
        inputs, kwargs = self._spec.separate_inputs(*args, **kwargs)
        execute_ws = workspace.get_workspace()
        for input, value in zip(self.inputs, inputs):
            if hasattr(value, 'id'):
                execute_ws.set_alias(value.id, input.id)
        return self.inputs, kwargs

    def __call__(self, *args, **kwargs):
        """Call the traced function."""
        if self.defs is None:
            inputs = []
            input_signature = self.input_signature
            with context.variable_scope('%s/Variable' % id(self)):
                for i in range(self._spec.num_inputs):
                    if input_signature is not None:
                        if i >= len(input_signature):
                            raise ValueError(
                                'When <example_inputs> is provided, '
                                'only define arguments covered by it.\n'
                                'Got %d inputs(s) and %d argument(s).'
                                % (len(input_signature), self._spec.num_inputs))
                        if input_signature[i] is not None:
                            inputs.append(Tensor(
                                *input_signature[i]['shape'],
                                dtype=input_signature[i]['dtype'],
                                device=input_signature[i]['device']))
                        else:
                            inputs.append(Tensor(1))
                    else:
                        inputs.append(Tensor(1))
                self.inputs = inputs
                with tape.GraphTape() as function_tape:
                    inputs, kwargs = self.separate_inputs(*args, **kwargs)
                    self.outputs = self._python_function(*inputs, **kwargs)
                    self.defs = function_tape.get_op_defs()
        else:
            self.separate_inputs(*args, **kwargs)
            workspace.get_workspace().run_operator(self.defs)
        return self.outputs

    def __get__(self, instance, owner):
        """Override to patch the instance methods."""
        del owner
        if instance not in self._descriptor_cache:
            if instance is None:
                return self
            self._descriptor_cache[instance] = \
                class_method_to_instance_method(self, instance)
        return self._descriptor_cache[instance]


def trace(func=None, example_inputs=None):
    """Trace a function and return an executable.

    Only the tensor operations could be traced:

    ```python
    def foo(x):
        return x + x

    bar = torch.jit.trace(foo, example_inputs=[torch.rand(1)])
    print(bar(torch.tensor([1, 2])))
    ```

    Above usages which can simplified as follows:

    ```python
    @torch.jit.trace(example_inputs=[torch.rand(1)])
    def foo(x):
        return x + x

    print(foo(torch.tensor([1, 2])))
    ```

    If providing ``nn.Module``, the ``forward`` method will be traced:

    ```python
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + x

    m = torch.jit.trace(MyModule(), example_inputs=[torch.rand(1)])
    print(m(torch.tensor([1, 2])))
    ```

    Parameters
    ----------
    func : Union[callable, dragon.vm.torch.nn.Module], required
        The function to be traced.
    example_inputs : Sequence[dragon.vm.torch.Tensor], required
        The examples to hint the input info.

    Returns
    -------
    callable
        A callable to execute the traced function.

    """
    def decorated(inner_function):
        if example_inputs is not None:
            input_signatures = []
            for inp in nest.flatten(example_inputs):
                if inp is not None:
                    input_signatures.append({
                        'shape': inp.shape,
                        'dtype': inp.dtype,
                        'device': inp.device})
                else:
                    input_signatures.append(None)
        else:
            input_signatures = None
        return decorator.make_decorator(
            inner_function, FunctionGuard(inner_function, input_signatures))
    if func is not None:
        if isinstance(func, Module):
            func.forward = decorated(func.forward)
            return func
        return decorated(func)
    return decorated
