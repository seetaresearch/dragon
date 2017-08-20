# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

try:
    import cPickle
except:
    import pickle as cPickle
import dragon.core.utils as utils
import dragon.core.mpi as mpi
import dragon.protos.dragon_pb2 as pb
import numpy as np
import os
from dragon import *
from dragon.config import logger
from google.protobuf.message import Message
from six.moves import range as xrange

CURRENT_GRAPH_IDX = 0

def StringfyProto(obj):
    if obj is str: return obj
    elif isinstance(obj, Message): return obj.SerializeToString()
    else: raise TypeError('object can not be serialized as a string')


def CreateGraph(graph_def):
    #PrintRawGraphDef(graph_def)
    CreateGraphCC(StringfyProto(graph_def))
    #PrintOptimizedGraph(graph_def)
    #WriteOptimizedGraph(graph_def)


def WriteOptimizedGraph(graph_def):
    with open(graph_def.name + '.txt', 'w') as f:
        f.write(str(graph_def))
        logger.info('write serialized graph to: {}'.format(graph_def.name + '.txt'))


def HasTensor(tensor):
    tensor = tensor.name if hasattr(tensor, 'name') else str(tensor)
    assert isinstance(tensor, str)
    return HasTensorCC(tensor)


def GetTensorName(tensor):
    tensor = tensor.name if hasattr(tensor, 'name') else str(tensor)
    assert isinstance(tensor, str)
    return GetTensorNameCC(tensor)


def CreateFiller(filler_def):
    filler_def = filler_def if isinstance(filler_def, str) \
        else filler_def.SerializeToString()
    CreateFillerCC(filler_def)


def FetchTensor(tensor):
    tensor = str(tensor.name) if hasattr(tensor, 'name') else str(tensor)
    assert isinstance(tensor, str)
    return FetchTensorCC(tensor)


def FeedTensor(tensor, ndarray, force_cpu=False, dtype=None):
    tensor = tensor.name if hasattr(tensor, 'name') else str(tensor)
    dev = None
    if force_cpu is True: dev = utils.MakeDeviceOption(0, 0)
    else:
        from dragon.core.scope import DEVICE_SCOPE
        if DEVICE_SCOPE != '':
            supports = {'/cpu': 0, '/gpu': 1}
            dev = pb.DeviceOption()
            dev.device_type = supports[DEVICE_SCOPE.split(':')[0]]
            dev.gpu_id = int(DEVICE_SCOPE.split(':')[1])
        else:
            from dragon.config import option
            if  option['device'] == 'CUDA':
                dev = utils.MakeDeviceOption(1, option['gpu_id'])
            elif option['device'] == 'CPU':
                dev = utils.MakeDeviceOption(0, 0)

    if not isinstance(ndarray, np.ndarray):
        if not isinstance(ndarray, list):
            ndarray = [ndarray]
        dtype = np.float32 if dtype is None else dtype
    else:
        dtype = ndarray.dtype if dtype is None else dtype
    ndarray = np.array(ndarray, dtype=dtype)
    FeedTensorCC(tensor, ndarray, StringfyProto(dev))


stages = {
    'forward': {'include': '', 'exclude': 'Gradient'},
    'backward': {'include': 'Gradient', 'exclude': 'Generate'},
    'backward_v2': {'include': 'Gradient', 'exclude': ''},
    'external_grads': {'include': '', 'exclude': 'Generate'}
}


def RunGraph(graph_name, inputs=(), outputs=[], stage=None, return_outputs=True):
    if len(inputs[0]) > 0:
        if len(inputs[0]) != len(inputs[1]):
            raise RuntimeError('function defined {} args, but only given {}'
                        .format(len(inputs[0]), len(inputs[1])))
        for idx in xrange(len(inputs[0])):
            FeedTensor(inputs[0][idx], inputs[1][idx])
    if stage is None: RunGraphCC(str(graph_name), '', '')
    else:
        state = stages[stage]
        RunGraphCC(str(graph_name), str(state['include']), str(state['exclude']))
    # force to return may lead crash if encountering un-computed outputs
    if return_outputs:
        if len(outputs) == 0 : return None
        elif len(outputs) == 1:  return outputs[0].get_value()
        else: return [outputs[i].get_value() for i in xrange(len(outputs))]


def PrintRawGraphDef(graph_def):
    logger.info(graph_def)


def PrintOptimizedGraph(graph_def):
    graph_name = graph_def.name
    graph_tensor = 'GraphDef_' + graph_name

    if not HasTensorCC(graph_tensor):
        logger.info('graph: {} does not exist, ignore printing....'.format(graph_name))
        return

    graph_def = pb.GraphDef()
    graph_def.ParseFromString(FetchTensor(graph_tensor))
    logger.info(graph_def)


def Snapshot(tensors, filename, prefix='', suffix='.bin', format=0):
    filepath = prefix + filename + suffix
    if mpi.is_init():
        if not mpi.allow_snapshot(): return
        filepath += '.rank.{}'.format(mpi.rank())

    dir = os.path.split(filepath)[0]
    if len(dir) > 0 and not os.path.exists(dir): os.makedirs(dir)

    if format is 0:
        # kv-store
        content = {}
        for tensor in tensors:
            content[tensor.name] = FetchTensor(tensor)
        with open(filepath, 'wb') as f:
            cPickle.dump(content, f, cPickle.HIGHEST_PROTOCOL)
        logger.info('Snapshot Model@: ' + filepath)
        logger.info('Model Format: cPickle')

    elif format is 1:
        # caffe-store
        names = [tensor.name for tensor in tensors]
        SnapshotCC(filepath, names, format)


def Restore(filename, format=0):
    if mpi.is_init():
        if not mpi.allow_snapshot():
            if not mpi.allow_parallel():
                filename += '.rank.{}'.format(mpi.rank())
                return

    assert os.path.exists(filename), 'model of path({}) does not exist.'.format(filename)
    if format is 0:
        content = cPickle.load(open(filename, 'rb'))
        logger.info('Restore From Model@: ' + filename)
        logger.info('Model Format: cPickle')
        for key, ndarray in content.items():
            if not HasTensor(key):
                logger.info('[Warning]:  Tensor({}) of model does not exist in any Graphs, skip.'.format(key))
            else:
                logger.info('[Info]: Tensor({}) restored.'.format(key))
                FeedTensor(key, ndarray)

    elif format is 1:
        # TODO(PhyscalX): caffemodel can't save the tensor name
        # TODO(PhyscalX): we simply use 'Scope + LayerName + @paramX'
        RestoreCC(filename, '', format)