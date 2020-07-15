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
"""Translate the graph abstraction to a python function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

from dragon.core.autograph import grad_maker
from dragon.core.autograph.op_def import OpDef
from dragon.core.autograph.op_def import OpInfo
from dragon.core.autograph.tensor import TensorRef
from dragon.core.framework import config
from dragon.core.framework import proto_util
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import logging
from dragon.core.util import nest


def add_device_option(graph_def):
    """Add the device option."""
    cfg = config.config()
    str2idx = {'cpu': 0, 'cuda': 1, 'cnml': 2}
    dev_opt = dragon_pb2.DeviceOption()
    dev_opt.device_type = str2idx[cfg.device_type]
    dev_opt.device_id = cfg.device_index
    dev_opt.random_seed = cfg.random_seed
    graph_def.device_option.CopyFrom(dev_opt)


def add_grad_info(graph_def, targets):
    """Add the gradient info."""
    for target in targets:
        info = target._grad
        if info is not None:
            graph_def.grad_info.extend([
                dragon_pb2.GradientInfo(
                    y=info.y.id,
                    xs=[x.id for x in info.xs])])


def add_optimization(graph_def, level=None):
    """Add the optimization argument."""
    cfg = config.config()
    if level is None:
        level = cfg.graph_optimization
    graph_def.arg.add().CopyFrom(
        proto_util.make_argument('optimization', level))
    graph_def.graph_type = cfg.graph_type


def add_phase(graph_def, targets):
    """Add the phase argument."""
    phase = 'TEST'
    for target in targets:
        try:
            if target._grad and target._grad.required():
                phase = 'TRAIN'
                break
        except AttributeError:
            pass
    graph_def.arg.extend([proto_util.make_argument('phase', phase)])


def add_update_defs(graph_def, optimizer):
    """Add the update defs."""
    if optimizer is None:
        return
    grads, update_defs = [], []
    extra_arguments = optimizer._extra_kwargs
    extra_arguments['handle'] = optimizer._op_handle
    # Generate op defs according to the collected updates
    current_ws = workspace.get_workspace()
    for (param, grad), arguments in optimizer._param_group:
        if current_ws.has_tensor(grad):
            grads.append(grad)
            arguments = dict(arguments, **extra_arguments)
            update_defs.append(
                proto_util.make_operator_def(
                    op_type=optimizer._op_type,
                    inputs=[grad],
                    outputs=[param],
                    name=OpDef.get_name(),
                    **arguments))
        else:
            logging.info('Skip to update Tensor({}).'.format(param))
    # Insert a reduce def if the process group is found.
    process_group = optimizer._process_group
    if process_group is not None:
        update_defs.insert(
            0, proto_util.make_operator_def(
                op_type='Collective',
                inputs=grads,
                outputs=grads,
                name=OpDef.get_name(),
                operation='MEAN',
                communication='ALLREDUCE',
                **process_group.arguments))
    graph_def.op.extend(update_defs)


class Function(object):
    """The class to compile graph into a callback function."""

    def __init__(self, name=None):
        self.callback = None
        self.graph_def = dragon_pb2.GraphDef()
        self.graph_def.name = name if name else 'Graph'
        self.graph_name = None  # Determined after creating
        self.inputs, self.outputs = None, None

    def create(self, inputs=None, outputs=None, givens=None, optimizer=None):
        self.inputs = inputs = [] if inputs is None else nest.flatten(inputs)
        self.outputs = outputs = [] if outputs is None else nest.flatten(outputs)

        if len(outputs) > 0 and optimizer is not None:
            raise ValueError('Specific either <outputs> or <optimizer>, not both.')

        # Collect the forward defs.
        op_info = OpInfo()
        requires_grad = False
        for i, output in enumerate(outputs):
            op_info.merge_from(output)
            op_info.add_target(output.id)
            try:
                grad_info = output._grad
                if grad_info and grad_info.required():
                    requires_grad = True
            except AttributeError:
                raise ValueError('Output[%d] is not a symbolic tensor.' % i)

        # Handle the replacements.
        if givens is not None:
            name_dict = {}
            for k, v in givens.items():
                if types.is_symbolic_tensor(v):
                    name_dict[k.id] = v.id
                    op_info.merge_from(v)
                else:
                    raise ValueError('Excepted a Tensor, got {}.'.format(type(v).__name__))
            # Update the original defs.
            op_info = copy.deepcopy(op_info)
            for k in op_info._defs.keys():
                op_def = op_info._defs[k]
                op_def.input.extend([
                    name_dict[input]
                    if input in name_dict else input
                    for input in op_def.input])
                del op_def.input[:len(op_def.input) // 2]

        # Sort out the forward defs.
        op_defs = sorted(op_info._defs.items(), key=lambda d: d[0])
        forward_defs = copy.deepcopy([v for k, v in op_defs])

        # Generate the backward defs.
        if requires_grad:
            input_grads, grad_targets = {}, []
            for output in outputs:
                info = output._grad
                if info is not None:
                    if info.grad_y is not None:
                        input_grads[output.id] = info.grad_y.id
                    grad_targets.append(output.id)
            backward_defs = grad_maker.GradientMaker.make(
                op_defs=forward_defs,
                targets=grad_targets,
                input_grads=input_grads,
            )
        else:
            backward_defs = []

        # Fill graph elements.
        self.graph_def.op.extend(forward_defs + backward_defs)
        self.graph_def.input.extend([input.name for input in inputs])
        self.graph_def.output.extend(list(op_info._targets))

        if len(outputs) > 0:
            add_device_option(self.graph_def)
            add_optimization(self.graph_def)
            add_grad_info(self.graph_def, outputs)
            add_phase(self.graph_def, outputs)
        elif optimizer is not None:
            add_device_option(self.graph_def)
            add_optimization(self.graph_def, level=0)
            add_update_defs(self.graph_def, optimizer)

        # Notify the backend to create and optimize.
        current_ws = workspace.get_workspace()
        self.graph_name = current_ws.create_graph(self.graph_def)

        # Bind a callback to run this graph.
        self.callback = lambda *args, **kwargs: \
            current_ws.run_graph(
                name=self.graph_name,
                inputs_and_values=(inputs, args),
                outputs=outputs,
                **kwargs
            )

        return self

    def export_to(self, name=None, export_dir='./'):
        """Export the graph into a text file.

        Parameters
        ----------
        name : str
            The optional graph name.
        export_dir : str
            The directory to export the file.

        """
        if not os.path.exists(export_dir):
            try:
                os.makedirs(export_dir)
            except Exception:
                raise ValueError('The given directory can not be created.')
        graph_def = copy.deepcopy(self.graph_def)
        graph_def.name = self.graph_def.name if name is None else name
        path = os.path.join(export_dir, graph_def.name + '.graph')
        with open(path, 'w') as f:
            f.write(str(graph_def))
        logging.info('Export meta graph into: {}'.format(path))

    def import_from(self, graph_def, explicit_inputs=False):
        """Import a defined function from a graph def.

        Set ``explicit_inputs`` to **True** to enforce feeding.

        Parameters
        ----------
        graph_def : GraphDef
            The definition of graph.
        explicit_inputs : bool
            Whether to enforce feeding on executing.

        Returns
        -------
        Function
            The self.

        """
        self.outputs = [TensorRef(name) for name in graph_def.output]
        self.inputs = [TensorRef(name).constant() for name in graph_def.input]

        # Fill with all known graph elements.
        add_device_option(graph_def)
        add_optimization(graph_def)
        add_phase(graph_def, self.outputs)

        # Notify the backend to create and optimize.
        current_ws = workspace.get_workspace()
        self.graph_def = graph_def
        self.graph_name = current_ws.create_graph(graph_def)

        # Bind a callback to run this graph.
        self.callback = lambda *args, **kwargs: \
            current_ws.run_graph(
                name=self.graph_name,
                inputs_and_values=(self.inputs if explicit_inputs else [], args),
                outputs=self.outputs,
                **kwargs
            )

        return self

    def __call__(self, *args, **kwargs):
        """Call the defined function."""
        return self.callback(*args, **kwargs) \
            if len(kwargs) > 0 else self.callback(*args)


def create_function(inputs=None, outputs=None, givens=None, optimizer=None):
    """Create a callable graph from specified outputs.

    Tensors that catch any operators can be used to create a graph:

    ```python
    x = dragon.Tensor(dtype='float32').constant()
    y = x * 2
    f = dragon.create_function(outputs=y)
    ```

    The created graph will be executed once the function is called:

    ```python
    x.set_value(numpy.ones((2, 3)))
    print(f())
    ```

    Specify ``inputs`` to feed values implicitly before graph executing:

    ```python
    f = dragon.create_function(inputs=x, outputs=y)
    print(f(numpy.ones((2, 3)))
    ```

    Specify ``givens`` to substitute tensors before creating:

    ```python
    x = dragon.Tensor(dtype='float32').constant()
    y = x * 2
    foo = dragon.create_function(outputs=y)

    # "bar" takes "x2" as input, and also writes to "y"
    x2 = dragon.Tensor(dtype='float32').constant()
    bar = dragon.create_function(outputs=y, givens={x: x2})
    ```

    Specify ``optimizer`` to make a graph applying parameter updates:

    ```python
    x = dragon.Tensor(dtype='float32').set_value(1)
    x_grad = dragon.Tensor(dtype='float32').set_value(1)

    optimizer = dragon.optimizers.SGD(base_lr=0.01)
    optimizer.apply_gradients(values_and_grads=[(x, x_grad)])

    # Compute x -= 0.01 * x_grad
    train_step = dragon.create_function(optimizer=optimizer)
    train_step()
    print(x.get_value())  # 0.99
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor], optional
        The input tensors.
    outputs : Sequence[dragon.Tensor], optional
        The output tensors.
    givens : Dict[dragon.Tensor, dragon.Tensor], optional
        The optional substitutions.
    optimizer : dragon.optimizers.Optimizer, optional
        The optional optimizer.

    Returns
    -------
    callable
        The callable function.

    """
    return Function().create(inputs, outputs, givens, optimizer)
