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

import copy
import numpy as np

import dragon.core.mpi as mpi
import dragon.core.workspace as ws
import dragon.protos.dragon_pb2 as pb

from dragon.core.utils import MakeArgument
from dragon.core.gradient_maker import GraphGradientMaker
from dragon.core.scope import GetOperatorName, GetTensorName
from dragon.core.tensor import Tensor


def GraphDef_Grad(meta_graph, targets):
    """Inject the gradient targets into GraphDef.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.
    targets : list
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
        for wrt in target.grad_wrts:
            all_pairs.add((target.name, wrt))

    for pair in all_pairs:
        g_target = pb.GradientTarget()
        g_target.cost = str(pair[0])
        g_target.wrt = str(pair[1])
        meta_graph.g_target.extend([g_target])


def GraphDef_Phase(meta_graph, targets):
    """Inject the phase into GraphDef.

    If existing gradients, we assume it should be ``TRAIN``, and vice versa.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.
    targets : list
        The solving targets.

    Returns
    -------
    None

    """
    phase = 'TEST'
    from dragon.core.scope import _PHASE_SCOPE
    if _PHASE_SCOPE != '':
        phase = _PHASE_SCOPE.upper()
    else:
        for target in targets:
            if len(target.grad_wrts) > 0:
                phase = 'TRAIN'
                break
    meta_graph.arg.extend([MakeArgument('phase', phase)])


def GraphDef_Update(meta_graph, updater):
    """Inject the update targets into GraphDef.

    The ``updater`` should generate update targets before.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.
    updater : BaseUpdater
        The updater.

    Returns
    -------
    None

    """
    if updater is None: return

    # use graph name if missing slot
    if updater._slot is None:
        updater._slot= meta_graph.name
    extra_arguments = updater._extra_kwargs
    extra_arguments['slot'] = updater._slot
    parallel_arguments = {}

    updater.register_in_workspace()

    # check data parallel if necessary
    if mpi.Is_Init():
        idx, group = mpi.AllowParallel()
        if idx != -1:
            parallel_arguments['parallel_mode'] = mpi.GetParallelMode()
            parallel_arguments['comm'], parallel_arguments['group'] \
                = mpi.CreateGroup(root=group[0], incl=group)
            parallel_arguments['root'] = group[0]
        for k, v in parallel_arguments.items():
            meta_graph.arg.add().CopyFrom(MakeArgument(k, v))

    for e in updater._param_group:
        pair, arguments = e
        kwargs = dict(arguments, **extra_arguments)
        u_target = pb.UpdateTarget()
        u_target.type = updater.type()
        _, u_target.name = GetOperatorName()
        for t in pair: u_target.tensor.append(t)
        for k, v in kwargs.items():
            u_target.arg.add().CopyFrom(MakeArgument(k, v))
        meta_graph.u_target.extend([u_target])


def GraphDef_Opt(meta_graph):
    """Inject the optimization options into GraphDef.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    References
    ----------
    `config.SetDebugMode(*args, **kwargs)`_ - How the enable debug mode.

    `memonger.share_grads(*args, **kwargs)`_ - How the enable gradients sharing.

    """

    from dragon.config import option
    OX = 3 if option['share_grads'] else 2
    if option['debug_mode']: OX = 1
    meta_graph.arg.add().CopyFrom(MakeArgument('optimization_level', OX))
    meta_graph.graph_type = option['graph_type']


def GraphDef_Device(meta_graph):
    """Inject the device option into GraphDef.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

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
        supports = {'CPU': 0, 'CUDA': 1, 'CNML': 2}
        device_option = pb.DeviceOption()
        device_option.device_type = supports[option['device']]
        device_option.device_id = option['device_id']
        device_option.random_seed = option['random_seed']
        if option['device'] == 'CUDA':
            if option['use_cudnn']: device_option.engine = 'CUDNN'
        meta_graph.device_option.CopyFrom(device_option)


def function(inputs=None, outputs=None, givens=None, updater=None):
    """Return a callable function that will compute ``outputs`` or apply ``updater``.

    Set ``inputs`` to feed inputs into this callable function.

    Set ``givens`` to substitute some tensors before making the computation graph.

    Set ``updater`` to make update graph, but the update targets should be generated before.

    Parameters
    ----------
    inputs : Tensor, list of Tensor or None
        The inputs to feed.
    outputs : Tensor, list of Tensor or None
        The outputs to solve.
    givens : dict or None
        The substitutions to use.
    updater : BaseUpdater
        The updater to use.

    Returns
    -------
    function
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

    all_exprs = dict()
    all_extra_targets = set()
    if not isinstance(outputs, list): outputs = [outputs]

    meta_graph = pb.GraphDef()

    meta_graph.name = 'Graph_' + str(ws.CURRENT_GRAPH_IDX)
    ws.CURRENT_GRAPH_IDX += 1

    # extract operators and targets from expressions
    existing_grads = False
    for output in outputs:
        meta_graph.target.extend([output.name])
        all_exprs.update(output.expressions)
        all_extra_targets = all_extra_targets.union(output.extra_targets)
        if len(output.grad_wrts) > 0: existing_grads = True

    # we should sort out the topology of these operators before using
    all_exprs = sorted(all_exprs.items(), key=lambda d: d[0])
    forward_ops = copy.deepcopy([v for k, v in all_exprs])

    # handle givens
    if givens is not None:
        name_dict = {}
        external_input_exprs = {}
        # extract new ops
        for old_tensor, new_tensor in givens.items():
            if isinstance(new_tensor, Tensor):
                name_dict[old_tensor.name] = new_tensor.name
                external_input_exprs.update(new_tensor.expressions)
            elif isinstance(new_tensor, np.ndarray):
                ws.FeedTensor(new_tensor, GetTensorName())
            all_extra_targets = all_extra_targets.union(new_tensor.extra_targets)
        external_input_exprs = sorted(external_input_exprs.items(), key=lambda d: d[0])
        external_input_ops = [v for k, v in external_input_exprs]
        # update original ops
        for op in forward_ops:
            op.input.extend([name_dict[input] if input in name_dict
                             else input for input in op.input])
            del op.input[:int(len(op.input) / 2)]
        # concat them together
        forward_ops = external_input_ops + forward_ops

    # handle grads
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
        meta_graph.target.extend([extra_target])

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

    # call c api to create graph
    ws.CreateGraph(meta_graph)

    # return a lambda point to run this graph
    return lambda *args, **kwargs: \
        ws.RunGraph(meta_graph.name, (inputs, args), outputs, **kwargs)


def eval(self, feed_dict=None):
    if not hasattr(self, '_eval_func'):
        if feed_dict is not None:
            self._eval_func = function(inputs=feed_dict.keys(), outputs=self)
        else:
            self._eval_func = function(outputs=self)

    # cond.1: run by feeding
    if feed_dict is not None:
        # checking
        for key, value in feed_dict.items():
            if not isinstance(key, Tensor):
                raise TypeError('The key of feed_dict key should be a Tensor.')
            if key.shape is not None:
                if len(key.shape) != len(value.shape):
                    raise RuntimeError(
                        'The Tensor({}) was limited to {} dimensions, \
                         while feed a value with {} dimensions.'.format(
                            key.name, len(key.shape), len(value.shape)))
                for i in range(len(key.shape)):
                    if key.shape[i] is None: continue
                    if key.shape[i] != value.shape[i]:
                        raise RuntimeError(
                            'The shape of Tensor({}) was limited as ('.format(key.name) +
                            ','.join([str(dim) for dim in key.shape]) + '), ' +
                            'while feed a value with (' + ','.join([str(dim) for dim in value.shape]) + ').')
        return self._eval_func(*feed_dict.values())
    else:
        # cond.2: run without feeding
        return self._eval_func()

Tensor.eval = eval