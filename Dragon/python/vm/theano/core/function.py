# --------------------------------------------------------
# Theano for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import copy
from collections import OrderedDict
import numpy as np
import sys

import dragon.core.mpi as mpi
import dragon.core.workspace as ws
import dragon.protos.dragon_pb2 as pb
from dragon.core.utils import MakeArgument
from dragon.core.gradient_maker import GraphGradientMaker
from dragon.core.scope import GetOperatorName, GetTensorName
from dragon.core.tensor import Tensor


def GraphDef_Grad(graph_def, targets):
    """ generate all graident targets for CC Graph """
    all_pairs = set()
    for target in targets:
        for wrt in target.grad_wrts:
            all_pairs.add((target.name, wrt))

    for pair in all_pairs:
        g_target = pb.GradientTarget()
        g_target.cost = str(pair[0])
        g_target.wrt = str(pair[1])
        graph_def.g_target.extend([g_target])

def GraphDef_Phase(graph_def, targets):
    phase = 'TEST'
    from dragon.core.scope import PHASE_SCOPE
    global PHASE_SCOPE
    if PHASE_SCOPE != '': phase = PHASE_SCOPE.upper()
    else:
        for target in targets:
            if len(target.grad_wrts) > 0:
                phase = 'TRAIN'
                break
    graph_def.arg.extend([MakeArgument('phase', phase)])

def GraphDef_Update(graph_def, updater):
    """ generate all update targets for CC Graph """
    if updater is None: return

    updater._prefix = graph_def.name + '_'
    extra_kwargs = updater._extra_kwargs
    extra_kwargs['domain'] = updater._prefix

    # wrap hyper-parameters as Tensor for CC
    for k,v in updater._hyper_params.items():
        ws.FeedTensor(updater._prefix + k, np.array([v], dtype=np.float32))

    # check data parallel if necessary
    if mpi.is_init():
        idx, group = mpi.allow_parallel()
        if idx != -1:
            extra_kwargs['comm'], extra_kwargs['group'] \
                = mpi.group(root=group[0], incl=group)
            extra_kwargs['root'] = group[0]
            extra_kwargs['mode'] = mpi.get_parallel_mode()
            extra_kwargs['group_size'] = len(group)

    for tuple in updater._tuples:
        tensors = tuple[0]; kwargs = tuple[1]
        kwargs = dict(kwargs, **extra_kwargs)
        u_target = pb.UpdateTarget()
        u_target.type = updater._type
        _, u_target.name = GetOperatorName()
        for tensor in tensors:
            u_target.tensor.append(tensor)
        for k,v in kwargs.items():
            u_target.arg.add().CopyFrom(MakeArgument(k, v))
        graph_def.u_target.extend([u_target])

def GraphDef_Debug(graph_def):
    """ generate debug mode for CC Graph """
    from dragon.config import option
    graph_def.debug_mode = option['debug_mode']

def GraphDef_Device(graph_def):
    """ generate deivce info for CC Graph """
    from dragon.config import option
    if option['device'] is not 'None':
        supports = {'CPU': 0, 'CUDA': 1}
        device_option = pb.DeviceOption()
        device_option.device_type = supports[option['device']]
        device_option.gpu_id = option['gpu_id']
        device_option.random_seed = option['random_seed']
        if option['use_cudnn']: device_option.engine = 'CUDNN'
        graph_def.device_option.CopyFrom(device_option)

def function(inputs=[], outputs=[], swaps=None, updater=None):
    """ return a excutable function for a graph """
    if not isinstance(inputs, list): inputs = [inputs]
    if not isinstance(outputs, list): outputs = [outputs]
    if len(outputs) > 0 and updater is not None:
        raise RuntimeError('outputs or updater must be in 2 function.')

    all_exprs = {}; all_extra_targets = set()
    if not isinstance(outputs, list): outputs = [outputs]

    graph_def = pb.GraphDef()

    graph_def.name = 'Graph_' + str(ws.CURRENT_GRAPH_IDX)
    ws.CURRENT_GRAPH_IDX += 1

    # extract operators and targets from expressions
    existing_grads = False
    for output in outputs:
        graph_def.target.extend([output.name])
        if sys.version_info >= (3, 0):
            all_exprs = OrderedDict(all_exprs, **output.expressions)
        else:
            all_exprs = dict(all_exprs, **output.expressions)
        all_extra_targets = all_extra_targets.union(output.extra_targets)
        if len(output.grad_wrts) > 0: existing_grads = True
    for extra_target in all_extra_targets: graph_def.target.extend([extra_target])

    # we should sort out the topology of these operators before using
    all_exprs = sorted(all_exprs.items(), key=lambda d:d[0])
    forward_ops = copy.deepcopy([v for k,v in all_exprs])

    # handle swap
    if swaps is not None:
        name_dict = {}
        external_input_exprs = {}

        for old_tenosr, new_tensor in swaps.items():
            if isinstance(new_tensor, Tensor):
                name_dict[old_tenosr.name] = new_tensor._name
                if sys.version_info >= (3, 0):
                    external_input_exprs = OrderedDict(external_input_exprs, **new_tensor.expressions)
                else:
                    external_input_exprs = dict(external_input_exprs, **new_tensor.expressions)
            elif isinstance(new_tensor, np.ndarray): ws.FeedTensor(new_tensor, GetTensorName())
        external_input_ops = [v for k,v in external_input_exprs.items()]
        for op in forward_ops:
            op.input.extend([name_dict[input] if input in name_dict
                                              else input for input in op.input])
            del op.input[:int(len(op.input)/2)]

        forward_ops = external_input_ops + forward_ops

    # handle grads
    if existing_grads:
        targets = [output.name for output in outputs]
        forward_ops, grad_ops = GraphGradientMaker.Make(forward_ops, targets)
    else: grad_ops = []
    graph_def.op.extend(forward_ops + grad_ops)

    if len(outputs) > 0:
        GraphDef_Device(graph_def)
        GraphDef_Debug(graph_def)
        GraphDef_Grad(graph_def, outputs)
        GraphDef_Phase(graph_def, outputs)

    elif updater is not None:
        GraphDef_Device(graph_def)
        GraphDef_Debug(graph_def)
        GraphDef_Update(graph_def, updater)

    # call c api to create graph
    ws.CreateGraph(graph_def)

    # return a lambda point to run this graph
    return lambda *args, **kwargs: \
        ws.RunGraph(graph_def.name, (inputs, args), outputs, **kwargs)