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

import os
import copy
import numpy as np

import dragon.core.mpi as mpi
import dragon.core.workspace as ws
import dragon.core.logging as logging
import dragon.proto.dragon_pb2 as pb

from dragon.core.proto_utils import MakeArgument
from dragon.core.helper import OperatorHelper
from dragon.core.gradient_maker import GraphGradientMaker
from dragon.core.scope import get_default_phase
from dragon.core.tensor import Tensor


def GraphDef_Grad(graph_def, targets):
    """Inject the gradient targets into GraphDef.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of graph.
    targets : sequence of Tensor
        The solving targets.

    Returns
    -------
    None

    See Also
    --------
    `T.grad(*args, **kwargs)`_ - How the generate gradient targets.

    """
    all_pairs = set()
    for target in targets:
        all_pairs.update(target.gradient.make_pairs())

    for pair in all_pairs:
        gradient = pb.GradientProto()
        gradient.cost, gradient.wrt = str(pair[0]), str(pair[1])
        graph_def.gradient.extend([gradient])


def GraphDef_Phase(graph_def, targets):
    """Inject the phase into GraphDef.

    If existing gradients, we assume it should be ``TRAIN``, and vice versa.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of graph.
    targets : sequence of Tensor
        The solving targets.

    Returns
    -------
    None

    """
    phase = get_default_phase()
    if phase is None:
        phase = 'TEST'
        for target in targets:
            if target.gradient.required():
                phase = 'TRAIN'
                break
    graph_def.arg.extend([MakeArgument('phase', phase)])


def GraphDef_Update(graph_def, updater):
    """Inject the update targets into GraphDef.

    The ``updater`` should generate update targets before.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of graph.
    updater : BaseUpdater
        The updater.

    Returns
    -------
    None

    """
    if updater is None: return

    extra_arguments = updater._extra_kwargs
    extra_arguments['slot'] = updater._slot
    parallel_arguments = {}

    updater.register_in_workspace()

    # Check data parallel if necessary
    if mpi.Is_Init():
        idx, group = mpi.AllowParallel()
        if idx != -1:
            parallel_arguments['parallel_mode'] = mpi.GetParallelMode()
            parallel_arguments['comm'], parallel_arguments['group'] \
                = mpi.CreateGroup(root=group[0], incl=group)
            parallel_arguments['root'] = group[0]
        for k, v in parallel_arguments.items():
            graph_def.arg.add().CopyFrom(MakeArgument(k, v))

    for e in updater._param_group:
        pair, arguments = e
        kwargs = dict(arguments, **extra_arguments)
        u_target = pb.UpdaterProto()
        u_target.type = updater.type()
        u_target.name = OperatorHelper.get_name()
        u_target.tensor.extend(pair)
        for k, v in kwargs.items():
            u_target.arg.add().CopyFrom(MakeArgument(k, v))
        graph_def.updater.extend([u_target])


def GraphDef_Opt(graph_def):
    """Inject the optimization options into GraphDef.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of graph.

    Returns
    -------
    None

    References
    ----------
    `config.SetDebugMode(*args, **kwargs)`_ - How the enable debug mode.

    `memonger.share_grads(*args, **kwargs)`_ - How the enable gradients sharing.

    """
    from dragon.config import option
    OX = option['graph_optimization_level']
    if not option['share_grads'] and OX >= 3: OX = 2
    graph_def.arg.add().CopyFrom(MakeArgument('optimization_level', OX))
    graph_def.graph_type = option['graph_type']


def GraphDef_Device(graph_def):
    """Inject the device option into GraphDef.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of graph.

    Returns
    -------
    None

    References
    ----------
    `config.EnableCPU()`_ - How to use CPU device.

    `config.EnableCUDA(*args, **kwargs)`_ - How to use CUDA device.

    `config.SetRandomSeed(*args, **kwargs)`_ - How to set random seed.

    """
    from dragon.config import option
    if option['device'] is not 'None':
        supports = {'cpu': 0, 'cuda': 1, 'cnml': 2}
        device_option = pb.DeviceOption()
        device_option.device_type = supports[option['device']]
        device_option.device_id = option['device_id']
        device_option.random_seed = option['random_seed']
        graph_def.device_option.CopyFrom(device_option)


class Function(object):
    """The ``Function`` wraps the meta graph and a defined callback.

    We recommend this way to avoid a explicit ``GraphDef`` exposed.

    """
    def __init__(self, name=None):
        self.callback = None
        self.meta_graph = pb.GraphDef()
        self.meta_graph.name = name if name else 'Graph'
        self.graph_name = None # Determined after creating

    def define(self, inputs=None, outputs=None, givens=None, updater=None):
        if not isinstance(inputs, list):
            if inputs is None:
                inputs = []
            else:
                inputs = [inputs]
        if not isinstance(outputs, list):
            if outputs is None:
                outputs = []
            else:
                outputs = [outputs]

        if len(outputs) > 0 and updater is not None:
            raise RuntimeError('You can specific either outputs or updater, not both.')

        all_expressions = dict()
        all_extra_targets = set()
        if not isinstance(outputs, list): outputs = [outputs]

        meta_graph = self.meta_graph

        # Extract operators and targets from expressions
        existing_grads = False
        for output in outputs:
            meta_graph.output.extend([output.name])
            all_expressions.update(output.expressions)
            all_extra_targets = all_extra_targets.union(output.extra_targets)
            if output.gradient.required(): existing_grads = True

        # We should sort out the topology of these operators before using
        all_expressions = sorted(all_expressions.items(), key=lambda d: d[0])
        forward_ops = copy.deepcopy([v for k, v in all_expressions])

        # Handle givens
        if givens is not None:
            name_dict = {}
            external_input_expressions = {}
            # Extract new ops
            for old_tensor, new_tensor in givens.items():
                if isinstance(new_tensor, Tensor):
                    name_dict[old_tensor.name] = new_tensor.name
                    external_input_expressions.update(new_tensor.expressions)
                else:
                    raise ValueError('Excepted a Tensor, '
                        'while got {}.'.format(type(new_tensor).__name__))
                all_extra_targets = all_extra_targets.union(new_tensor.extra_targets)
            external_input_expressions = sorted(external_input_expressions.items(), key=lambda d: d[0])
            external_input_ops = [v for k, v in external_input_expressions]
            # Update original ops
            for op in forward_ops:
                op.input.extend([name_dict[input] if input in name_dict
                                 else input for input in op.input])
                del op.input[:int(len(op.input) / 2)]
            # Concat them together
            forward_ops = external_input_ops + forward_ops

        # Handle grads
        if existing_grads:
            targets = [output.name for output in outputs]
            targets.extend(all_extra_targets)
            forward_ops, grad_ops, _ = \
                GraphGradientMaker.Make(forward_ops, targets)
        else:
            grad_ops = []

        # Write Ops
        meta_graph.op.extend(forward_ops + grad_ops)

        # Write Extra Targets
        for extra_target in all_extra_targets:
            meta_graph.output.extend([extra_target])

        # Write External Inputs
        for input in inputs:
            meta_graph.input.extend([input.name])

        self.inputs, self.outputs = inputs, outputs

        # Write Misc
        if len(outputs) > 0:
            GraphDef_Device(meta_graph)
            GraphDef_Opt(meta_graph)
            GraphDef_Grad(meta_graph, outputs)
            GraphDef_Phase(meta_graph, outputs)

        elif updater is not None:
            GraphDef_Device(meta_graph)
            GraphDef_Opt(meta_graph)
            GraphDef_Update(meta_graph, updater)

        # Call c api to create graph
        self.graph_name = ws.CreateGraph(meta_graph)

        # Bind a lambda callback to run this graph
        self.callback = lambda *args, **kwargs: \
            ws.RunGraph(self.graph_name, (inputs, args), outputs, **kwargs)

        # Self return
        return self

    def export_to(self, name=None, export_dir='./'):
        """Export the meta graph of this defined function.

        Parameters
        ----------
        export_dir : str
            The directory to export the meta text file.

        Returns
        -------
        None

        """
        if not os.path.exists(export_dir):
            try:
                os.makedirs(export_dir)
            except Exception:
                raise ValueError('The given directory can not be created.')
        meta_graph_copy = copy.deepcopy(self.meta_graph)
        meta_graph_copy.name = self.meta_graph.name if name is None else name
        file = os.path.join(export_dir, meta_graph_copy.name + '.metatxt')
        with open(file, 'w') as f: f.write(str(meta_graph_copy))
        logging.info('Export meta graph into: {}'.format(file))

    def import_from(self, graph_def, explicit_inputs=False):
        """Import the defined function from a graph def.

        Set ``explicit_inputs`` to ``False``,

        if you want to feed the inputs as ``workspace.FeedTensor(self.inputs)``

        Parameters
        ----------
        meta_graph : GraphDef
            The definition of graph.
        explicit_inputs : boolean
            Whether to enforce feeding on executing.

        Returns
        -------
        Function
            The self.

        """
        self.inputs = [Tensor(name=input).Variable() for input in graph_def.input]
        self.outputs = [Tensor(name=output) for output in graph_def.output]

        GraphDef_Device(graph_def)
        GraphDef_Opt(graph_def)
        GraphDef_Phase(graph_def, self.outputs)

        # Store for future development
        self.meta_graph = graph_def

        # Call c api to create graph
        self.graph_name = ws.CreateGraph(graph_def)

        # Bind a lambda callback to run this graph
        callback_inputs = self.inputs if explicit_inputs else []
        self.callback = lambda *args, **kwargs: \
            ws.RunGraph(self.graph_name, (callback_inputs, args), self.outputs, **kwargs)

        # Self return
        return self

    def __call__(self, *args, **kwargs):
        """Call the defined function."""
        return self.callback(*args, **kwargs) \
            if len(kwargs) > 0 else self.callback(*args)


def function(inputs=None, outputs=None, givens=None, updater=None):
    """Return a callable function that will compute ``outputs`` or apply ``updater``.

    Set ``inputs`` to feed inputs into this callable function.

    Set ``givens`` to substitute some tensors before making the computation graph.

    Set ``updater`` to make update graph, but the update targets should be generated before.

    Parameters
    ----------
    inputs : sequence of Tensor, optional
        The inputs to feed.
    outputs : sequence of Tensor, optional
        The outputs to fetch.
    givens : dict of Tensor, optional
        The substitutions to use.
    updater : Updater, optional
        The updater to use.

    Returns
    -------
    Function
        The callable function.

    Examples
    --------
    >>> x = Tensor('x', dtype='float32').Variable()
    >>> y = x * 2
    >>> f = function(outputs=y)
    >>> x.set_value(np.ones((2, 3)))
    >>> print(f())
    >>> [[ 2.  2.  2.]
         [ 2.  2.  2.]]

    >>> f = function(inputs=x, outputs=y)
    >>> print(f(np.ones((2, 3)))
    >>> [[ 2.  2.  2.]
         [ 2.  2.  2.]]

    """
    return Function().define(inputs, outputs, givens, updater)