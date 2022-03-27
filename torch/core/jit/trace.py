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

from dragon.core.autograph import function_lib
from dragon.core.framework import context
from dragon.core.framework import tapes
from dragon.core.framework import workspace
from dragon.core.util import decorator
from dragon.core.util import nest
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.tensor import Tensor


class FunctionGuard(function_lib.FunctionGuard):
    """Map the run function to workspace-local functions."""

    def _trace_operators(self, *args, **kwargs):
        attributes = self._attribute_cache[workspace.get_workspace()]
        input_signature = self._spec.input_signature
        args, kwargs = self._spec.separate_inputs(*args, **kwargs)
        inputs = []
        with context.variable_scope('%s/Variable' % id(self)):
            for i in range(self._spec.num_inputs):
                input, input_spec = args[i], None
                if input_signature is not None:
                    input_spec = input_signature[i]
                if not isinstance(input, Tensor) and input_spec is None:
                    inputs.append(input)
                    continue
                input_spec = input_spec or {}
                for k in ('shape', 'dtype', 'device'):
                    input_spec[k] = getattr(input, k, input_spec.get(k, None))
                inputs.append(Tensor(*input_spec['shape'],
                                     dtype=input_spec['dtype'],
                                     device=input_spec['device']))
                if isinstance(input, Tensor):
                    inputs[-1].copy_(input)
            with tapes.Tape() as function_tape:
                function_tape._tracing = True
                attributes['inputs'] = inputs
                attributes['outputs'] = self._run_function(*inputs, **kwargs)
                attributes['operators'] = function_tape.get_elements()
        return attributes['operators']

    def __call__(self, *args, **kwargs):
        """Call the traced function."""
        execute_ws = workspace.get_workspace()
        attributes = self._attribute_cache[execute_ws]
        operators = attributes.get('operators', None)
        if operators is None:
            operators = self._trace_operators(*args, **kwargs)
        inputs = attributes['inputs']
        values, _ = self._spec.separate_inputs(*args, **kwargs)
        for input, value in zip(inputs, values):
            if not isinstance(input, Tensor):
                continue
            if hasattr(value, 'id'):
                execute_ws.set_alias(value.id, input.id)
        execute_ws.run_operator(operators)
        return attributes['outputs']


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
