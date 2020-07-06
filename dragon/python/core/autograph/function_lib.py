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
from dragon.core.autograph.tensor import Tensor
from dragon.core.framework import config
from dragon.core.framework import context
from dragon.core.framework import proto_util
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import logging
from dragon.core.util import nest


def add_device_option(graph_def):
    """Add the device option for graph."""
    cfg = config.config()
    str2idx = {'cpu': 0, 'cuda': 1, 'cnml': 2}
    dev_opt = dragon_pb2.DeviceOption()
    dev_opt.device_type = str2idx[cfg.device_type]
    dev_opt.device_id = cfg.device_index
    dev_opt.random_seed = cfg.random_seed
    graph_def.device_option.CopyFrom(dev_opt)


def add_gradient_info(graph_def, targets):
    """Add the gradient info for graph."""
    gradients = set()
    for target in targets:
        if target._grad is not None:
            gradients.update(target._grad.make_pairs())
    for (cost, wrt) in gradients:
        gradient = dragon_pb2.GradientProto()
        gradient.cost, gradient.wrt = str(cost), str(wrt)
        graph_def.gradient.extend([gradient])


def add_optimization(graph_def, level=None):
    """Add the optimization attribute for graph."""
    cfg = config.config()
    if level is None:
        level = cfg.graph_optimization
    graph_def.arg.add().CopyFrom(
        proto_util.make_argument(
            'optimization_level', level))
    graph_def.graph_type = cfg.graph_type


def add_phase(graph_def, targets):
    """Add the phase attribute for graph."""
    phase = context.get_graph_phase()
    if phase is None:
        phase = 'TEST'
        for target in targets:
            if target._grad is not None and \
                    target._grad.required():
                phase = 'TRAIN'
                break
    graph_def.arg.extend([proto_util.make_argument('phase', phase)])


def add_update_ops(graph_def, optimizer):
    """Add the update operators for graph."""
    if optimizer is None:
        return
    grads, update_ops = [], []
    extra_arguments = optimizer._extra_kwargs
    extra_arguments['handle'] = optimizer._op_handle
    # Generate update operators according to the updater.
    for e in optimizer._param_group:
        (param, grad), arguments = e
        if workspace.has_tensor(grad):
            grads.append(grad)
            arguments = dict(arguments, **extra_arguments)
            update_ops.append(
                proto_util.make_operator_def(
                    op_type=optimizer._op_type,
                    inputs=[grad],
                    outputs=[param],
                    name=OpDef.get_name(),
                    **arguments
                ))
        else:
            logging.info('Skip to update Tensor({}).'.format(param))
    # Insert a reduce op if the process group is found.
    process_group = optimizer._process_group
    if process_group is not None:
        update_ops.insert(
            0, proto_util.make_operator_def(
                op_type='Collective',
                inputs=grads,
                outputs=grads,
                name=OpDef.get_name(),
                operation='MEAN',
                communication='ALLREDUCE',
                **process_group.arguments
            )
        )
    graph_def.op.extend(update_ops)


class Function(object):
    """The class to compile graph into a callback function."""

    def __init__(self, name=None):
        self.callback = None
        self.graph_def = dragon_pb2.GraphDef()
        self.graph_def.name = name if name else 'Graph'
        self.graph_name = None  # Determined after creating
        self.inputs, self.outputs = None, None

    def create(self, inputs=None, outputs=None, givens=None, updater=None):
        self.inputs = inputs = [] if inputs is None else nest.flatten(inputs)
        self.outputs = outputs = [] if outputs is None else nest.flatten(outputs)

        if len(outputs) > 0 and updater is not None:
            raise ValueError('Specific either <outputs> or <updater>, not both.')

        op_info = OpInfo()

        # Collect the forward operators.
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

        # Handle givens.
        if givens is not None:
            name_dict = {}
            for k, v in givens.items():
                if isinstance(v, Tensor):
                    name_dict[k.id] = v.id
                    op_info.merge_from(v)
                else:
                    raise ValueError(
                        'Excepted a Tensor, '
                        'got {}.'.format(type(v).__name__)
                    )
            # Update original operators.
            op_info = copy.deepcopy(op_info)
            for k in op_info._defs.keys():
                op_def = op_info._defs[k]
                op_def.input.extend([
                    name_dict[input]
                    if input in name_dict else input
                    for input in op_def.input
                ])
                del op_def.input[:len(op_def.input) // 2]

        # Sort out the states.
        op_defs = sorted(op_info._defs.items(), key=lambda d: d[0])
        forward_ops = copy.deepcopy([v for k, v in op_defs])

        # Generate the backward operators.
        if requires_grad:
            input_grads, grad_targets = {}, []
            for output in outputs:
                grad_info = output._grad
                if grad_info is not None:
                    if grad_info.input is not None:
                        input_grads[output.id] = output._grad.input.id
                    grad_targets.append(output.id)
            forward_ops, gradient_ops, _ = \
                grad_maker.GradientMaker.make(
                    forward_ops=forward_ops,
                    targets=grad_targets,
                    input_grads=input_grads,
                )
        else:
            gradient_ops = []

        # Fill with all known graph elements.
        self.graph_def.op.extend(forward_ops + gradient_ops)
        self.graph_def.input.extend([input.name for input in inputs])
        self.graph_def.output.extend(list(op_info._targets))

        if len(outputs) > 0:
            add_device_option(self.graph_def)
            add_optimization(self.graph_def)
            add_gradient_info(self.graph_def, outputs)
            add_phase(self.graph_def, outputs)
        elif updater is not None:
            add_device_option(self.graph_def)
            add_optimization(self.graph_def, level=0)
            add_update_ops(self.graph_def, updater)

        # Notify the backend to create and optimize.
        self.graph_name = workspace.create_graph(self.graph_def)

        # Bind a callback to run this graph.
        self.callback = lambda *args, **kwargs: \
            workspace.run_graph(
                graph=self.graph_name,
                inputs=(inputs, args),
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
        self.outputs = [Tensor(name) for name in graph_def.output]
        self.inputs = [Tensor(name).variable() for name in graph_def.input]

        # Fill with all known graph elements.
        add_device_option(graph_def)
        add_optimization(graph_def)
        add_phase(graph_def, self.outputs)

        # Notify the backend to create and optimize.
        self.graph_def = graph_def
        self.graph_name = workspace.create_graph(graph_def)

        # Bind a callback to run this graph.
        callback_inputs = self.inputs if explicit_inputs else []
        self.callback = lambda *args, **kwargs: \
            workspace.run_graph(
                graph=self.graph_name,
                inputs=(callback_inputs, args),
                outputs=self.outputs,
                **kwargs
            )

        return self

    def __call__(self, *args, **kwargs):
        """Call the defined function."""
        return self.callback(*args, **kwargs) \
            if len(kwargs) > 0 else self.callback(*args)


def create_function(inputs=None, outputs=None, givens=None, updater=None):
    """Create a callable graph from specified outputs.

    Tensors that catch any operators can be used to create a graph:

    ```python
    x = dragon.Tensor('x', dtype='float32').variable()
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
    x = dragon.Tensor('x', dtype='float32').variable()
    y = x * 2
    foo = dragon.create_function(outputs=y)

    # "bar" takes "x2" as input, and also writes to "y"
    x2 = dragon.Tensor('x2', dtype='float32').variable()
    bar = dragon.create_function(outputs=y, givens={x: x2})
    ```

    Specify ``updater`` to make a graph applying SGD updates:

    ```python
    x = dragon.Tensor('x', dtype='float32').set_value(1)
    x_grad = dragon.Tensor('x_grad', dtype='float32').set_value(1)

    # Define a updater to catch the operators
    updater = dragon.updaters.SGD(base_lr=0.01)
    updater.apply_gradients(values_and_grads=[(x, x_grad)])

    # Compute x -= 0.01 * x_grad
    train_step = dragon.create_function(updater=updater)
    train_step()
    print(x.get_value())
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor], optional
        The inputs to feed.
    outputs : Sequence[dragon.Tensor], optional
        The outputs to fetch.
    givens : Dict[dragon.Tensor, dragon.Tensor], optional
        The substitutions to apply.
    updater : Updater, optional
        The optional updater.

    Returns
    -------
    Function
        The callable function.

    """
    return Function().create(inputs, outputs, givens, updater)
