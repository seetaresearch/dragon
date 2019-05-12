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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

from dragon import config as _cfg
from dragon.core.tensor import Tensor as _Tensor
from dragon.core import mpi as _mpi
from dragon.core import scope as _scope
from dragon.core import helper as _helper
from dragon.core import logging as _logging
from dragon.core import workspace as _workspace
from dragon.proto import dragon_pb2 as _proto_def
from dragon.core import proto_utils as _proto_utils
from dragon.core import gradient_maker as _gradient_maker


def _inject_gradients(graph_def, targets):
    """Inject the gradients into GraphDef.

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
    gradients = set()
    for target in targets:
        gradients.update(target.gradient.make_pairs())

    for (cost, wrt) in gradients:
        gradient = _proto_def.GradientProto()
        gradient.cost, gradient.wrt = str(cost), str(wrt)
        graph_def.gradient.extend([gradient])


def _inject_phase(graph_def, targets):
    """Inject the phase info into GraphDef.

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
    phase = _scope.get_default_phase()
    if phase is None:
        phase = 'TEST'
        for target in targets:
            if target.gradient.required():
                phase = 'TRAIN'
                break
    graph_def.arg.extend([
        _proto_utils.MakeArgument(
            'phase', phase)])


def _inject_update_ops(graph_def, updater):
    """Inject the update ops GraphDef.

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
    updater.register_in_workspace()

    grads, update_ops = [], []
    extra_arguments = updater._extra_kwargs
    extra_arguments['slot'] = updater._slot

    # Build update ops according to the updater
    for e in updater._param_group:
        (param, grad), arguments = e
        if _workspace.HasTensor(grad):
            grads.append(grad)
            arguments = dict(arguments, **extra_arguments)
            update_ops.append(
                _proto_utils.
                    MakeOperatorDef(
                        op_type=updater.type(),
                        inputs=[grad],
                        outputs=[param],
                        name=_helper.OperatorHelper.get_name(),
                        **arguments
                    )
                )
        else:
            _logging.info('Skip to update Tensor({}).'.format(param))

    # Check data parallel if necessary
    if _mpi.Is_Init():
        (rank, group), arguments = _mpi.AllowParallel(), {}
        if rank != -1:
            arguments['mode'] = '%s_ALLREDUCE' % _mpi.GetParallelMode()
            arguments['root'], (arguments['comm'], arguments['group']) \
                = group[0], _mpi.CreateGroup(root=group[0], incl=group)
            update_ops.insert(
                0, _proto_utils.
                    MakeOperatorDef(
                        op_type='CollectiveUpdate',
                        inputs=grads,
                        outputs=grads,
                        name=_helper.OperatorHelper.get_name(),
                        **arguments
                    )
                )

    graph_def.op.extend(update_ops)


def _inject_optimization(graph_def, opt_level=None):
    """Inject the optimization info into GraphDef.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of graph.
    opt_level : int, optional
        The optimization level.

    Returns
    -------
    None

    References
    ----------
    `config.SetDebugMode(*args, **kwargs)`_ - How the enable debug mode.

    `memonger.share_grads(*args, **kwargs)`_ - How the enable gradients sharing.

    """
    options = _cfg.GetGlobalOptions()
    if opt_level is None:
        opt_level = options['graph_optimization_level']
        if not options['share_grads'] and \
            opt_level >= 3: opt_level = 2
    graph_def.arg.add().CopyFrom(
        _proto_utils.MakeArgument(
            'optimization_level', opt_level))
    graph_def.graph_type = options['graph_type']


def _inject_device(graph_def):
    """Inject the device info into GraphDef.

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
    options = _cfg.GetGlobalOptions()
    if options['device'] is not 'none':
        supports = {'cpu': 0, 'cuda': 1, 'cnml': 2}
        device_option = _proto_def.DeviceOption()
        device_option.device_type = supports[options['device']]
        device_option.device_id = options['device_id']
        device_option.random_seed = options['random_seed']
        graph_def.device_option.CopyFrom(device_option)


class Function(object):
    """The ``Function`` wraps the meta graph and a defined callback.

    We recommend this way to avoid a explicit ``GraphDef`` exposed.

    """
    def __init__(self, name=None):
        self.callback = None
        self.meta_graph = _proto_def.GraphDef()
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
                if isinstance(new_tensor, _Tensor):
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
                _gradient_maker.GraphGradientMaker \
                    .Make(forward_ops, targets)
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

        # Inject arguments based on global options
        if len(outputs) > 0:
            _inject_device(meta_graph)
            _inject_optimization(meta_graph)
            _inject_gradients(meta_graph, outputs)
            _inject_phase(meta_graph, outputs)

        elif updater is not None:
            _inject_device(meta_graph)
            _inject_optimization(meta_graph, opt_level=0)
            _inject_update_ops(meta_graph, updater)

        # Call c api to create graph
        self.graph_name = _workspace.CreateGraph(meta_graph)

        # Bind a lambda callback to run this graph
        self.callback = lambda *args, **kwargs: \
            _workspace.RunGraph(
                graph_name=self.graph_name,
                    inputs=(inputs, args),
                        outputs=outputs, **kwargs)

        # Return the self
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
        _logging.info('Export meta graph into: {}'.format(file))

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
        self.inputs = [_Tensor(input).Variable() for input in graph_def.input]
        self.outputs = [_Tensor(output) for output in graph_def.output]

        _inject_device(graph_def)
        _inject_optimization(graph_def)
        _inject_phase(graph_def, self.outputs)

        # Store for future development
        self.meta_graph = graph_def

        # Call c api to create graph
        self.graph_name = _workspace.CreateGraph(graph_def)

        # Bind a lambda callback to run this graph
        callback_inputs = self.inputs if explicit_inputs else []
        self.callback = lambda *args, **kwargs: \
            _workspace.RunGraph(
                self.graph_name,
                    (callback_inputs, args),
                        self.outputs, **kwargs)

        # Return self
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
    >>> import numpy, dragon
    >>> x = dragon.Tensor('x', dtype='float32').Variable()
    >>> y = x * 2
    >>> f = function(outputs=y)
    >>> x.set_value(numpy.ones((2, 3)))
    >>> print(f())
    >>> [[ 2.  2.  2.]
         [ 2.  2.  2.]]

    >>> f = function(inputs=x, outputs=y)
    >>> print(f(numpy.ones((2, 3)))
    >>> [[ 2.  2.  2.]
         [ 2.  2.  2.]]

    """
    return Function().define(inputs, outputs, givens, updater)