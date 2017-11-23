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
from google.protobuf.message import Message
from six.moves import range as xrange

CURRENT_GRAPH_IDX = 0

__all__ = [
    'SwitchWorkspace',
    'CreateGraph',
    'RunGraph',
    'HasTensor',
    'FeedTensor',
    'FetchTensor',
    'GetTensorName',
    'CreateFiller',
    'Snapshot',
    'Restore',
    'LogMetaGraph',
    'LogOptimizedGraph',
    'ExportMetaGraph'
]

_DATA_TYPES = {
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

def _stringify_proto(obj):
    """
    Stringify a protobuf structure.
    """
    if obj is str: return obj
    elif isinstance(obj, Message): return obj.SerializeToString()
    else: raise TypeError('object can not be serialized as a string')


def SwitchWorkspace(workspace, create_if_missing=True):
    """Switch to the specific workspace.

    Parameters
    ----------
    workspace : str
        The name of the specific workspace.
    create_if_missing : boolean
        Whether to create the specific workspace if it does not exist.

    Returns
    -------
    None

    References
    ----------
    The wrapper of ``SwitchWorkspaceCC``.

    """
    SwitchWorkspaceCC(workspace, create_if_missing)


def CreateGraph(meta_graph):
    """Create the graph in the VM backend.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    References
    ----------
    The wrapper of ``CreateGraphCC``.

    """
    LogMetaGraph(meta_graph)
    ExportMetaGraph(meta_graph)
    CreateGraphCC(_stringify_proto(meta_graph))
    LogOptimizedGraph(meta_graph)


def HasTensor(tensor):
    """Query whether tensor has registered in current workspace.

    Parameters
    ----------
    tensor : Tensor
        The tensor to query.

    Returns
    -------
    boolean
        The query result.

    References
    ----------
    The wrapper of ``HasTensorCC``.

    """
    tensor = tensor.name if hasattr(tensor, 'name') else str(tensor)
    assert isinstance(tensor, str)
    return HasTensorCC(tensor)


def GetTensorName(tensor):
    """Query the name represented in current workspace.

    Parameters
    ----------
    tensor : Tensor
        The tensor to query.

    Returns
    -------
    str
        The real(inplace-optimized) name in the backend.

    Notes
    -----
    The query result may be different from the one used in the frontend.

    References
    ----------
    The wrapper of ``GetTensorNameCC``.

    """
    tensor = tensor.name if hasattr(tensor, 'name') else str(tensor)
    assert isinstance(tensor, str)
    return GetTensorNameCC(tensor)


def CreateFiller(filler_def):
    """Create the filler in the backend.

    Parameters
    ----------
    filler_def : dragon_pb2.TensorFiller
        The

    Returns
    -------
    None

    See Also
    --------
    `Tensor.Fill(*args, **kwargs)
    <tensor.html#dragon.core.tensor.Tensor.Fill>`_ - How to fill a Tensor. [**Caffe Style**]

    References
    ----------
    The wrapper of ``CreateFillerCC``.

    """
    filler_def = filler_def if isinstance(filler_def, str) \
        else filler_def.SerializeToString()
    CreateFillerCC(filler_def)


def FetchTensor(tensor):
    """Fetch the values of given tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor to fetch.

    Returns
    -------
    numpy.ndarray
        The values copied from the backend.

    References
    ----------
    The wrapper of ``FetchTensorCC``.

    """
    tensor = str(tensor.name) if hasattr(tensor, 'name') else str(tensor)
    assert isinstance(tensor, str)
    return FetchTensorCC(tensor)


def FeedTensor(tensor, ndarray, force_cpu=False, dtype=None):
    """Feed the values to the given tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to feed.
    ndarray : basic type, list or numpy.ndarray
        The values to feed.
    force_cpu : boolean
        Whether force to feed to cpu context.
    dtype : np.dtype or None
        The data type. If ``None``, np.float32 will be used instead.

    Returns
    -------
    None

    Examples
    --------
    >>> import dragon.core.workspace as ws
    >>> a = Tensor().Variable()
    >>> ws.FeedTensor(a, 1)
    >>> a_value = ws.FetchTensor(a)
    >>> a_value, a_value.dtype
    >>> [ 1.], float32

    >>> ws.FeedTensor(a, [[1, 2, 3]], dtype=np.float16)
    >>> a_value = a.get_value()
    >>> a_value, a_value.dtype
    >>> [[ 1.  2.  3.]], float16

    References
    ----------
    The wrapper of ``FeedTensorCC``.

    """
    name = tensor.name if hasattr(tensor, 'name') else str(tensor)
    dev = None
    if force_cpu is True: dev = utils.MakeDeviceOption(0, 0)
    else:
        from dragon.core.scope import _DEVICE_SCOPE
        if _DEVICE_SCOPE != '':
            supports = {'/cpu': 0, '/gpu': 1}
            dev = pb.DeviceOption()
            dev.device_type = supports[_DEVICE_SCOPE.split(':')[0]]
            dev.gpu_id = int(_DEVICE_SCOPE.split(':')[1])
        else:
            from dragon.config import option
            if  option['device'] == 'CUDA':
                dev = utils.MakeDeviceOption(1, option['gpu_id'])
            elif option['device'] == 'CPU':
                dev = utils.MakeDeviceOption(0, 0)

    if not isinstance(ndarray, np.ndarray):
        if not isinstance(ndarray, list):
            ndarray = [ndarray]
        auto_dtype = np.float32 if dtype is None else dtype
    else:
        auto_dtype = ndarray.dtype if dtype is None else dtype

    if hasattr(tensor, 'dtype') and tensor.dtype is not None:
        if tensor.dtype not in _DATA_TYPES:
            raise TypeError('Unsupported data types: {}.'.format(tensor.dtype))
        preset_dtype = _DATA_TYPES[tensor.dtype]
        if dtype is not None:
            if dtype != preset_dtype:
                raise TypeError('The preset data type is {}, but force to {}.'.
                                format(preset_dtype, dtype))
        auto_dtype = preset_dtype
    ndarray = np.array(ndarray, dtype=auto_dtype)
    FeedTensorCC(name, ndarray, _stringify_proto(dev))


stages = {
    'forward': {'include': '', 'exclude': 'Gradient'},
    'backward': {'include': 'Gradient', 'exclude': 'Generate'},
    'backward_v2': {'include': 'Gradient', 'exclude': ''},
    'external_grads': {'include': '', 'exclude': 'Generate'}
}


def RunGraph(graph_name, inputs=(), outputs=[], stage=None, return_outputs=True):
    """Run the specific graph.

    Parameters
    ----------
    graph_name : str
        The name of the graph.
    inputs : tuple
        The tensors(list) and corresponding values(list).
    outputs : list of Tensor
        The outputs of the graph.
    stage : str
        The preset custom stages. See ``stages``.
    return_outputs : boolean
        Whether to return the outputs.

    Returns
    -------
    None, numpy.ndarry or list of numpy.ndarray
        The outputs, format as numpy.ndarray.

    See Also
    --------
    `theano.function(*args, **kwargs)`_ - How to make a graph. [**Theano Style**]

    """
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


def LogMetaGraph(meta_graph):
    """Log the meta graph.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    """
    from dragon.config import option, logger
    if option['log_meta_graph']:
        logger.info(meta_graph)


def GetOptimizedGraph(meta_graph):
    """Return the optimized graph.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

    Returns
    -------
    graph_def : dragon_pb2.GraphDef
        The definition of optimized graph.

    """
    from dragon.config import logger
    graph_name = meta_graph.name
    graph_tensor = 'GraphDef_' + graph_name

    if not HasTensorCC(graph_tensor):
        logger.info('Graph({}) does not exist, ignore printing....'.format(graph_name))
        return

    opt_graph_def = pb.GraphDef()
    opt_graph_def.ParseFromString(FetchTensor(graph_tensor))
    return opt_graph_def


def LogOptimizedGraph(meta_graph):
    """Log the optimized graph.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    """
    from dragon.config import option, logger
    if option['log_optimized_graph']:
        optimized_graph = GetOptimizedGraph(meta_graph)
        logger.info(optimized_graph)


def ExportMetaGraph(meta_graph):
    """Export the meta graph into a file under specific folder.

    You can set the exporting prefix by `config.ExportMetaGraph(prefix)`_.

    Parameters
    ----------
    meta_graph : dragon_pb2.GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    """
    from dragon.config import option, logger
    if option['export_meta_graph']:
        if not os.path.exists(option['export_meta_graph']):
            try:
                os.makedirs(option['export_meta_graph'])
            except Exception:
                raise ValueError('The given prefix is invalid.')
        filepath = os.path.join(option['export_meta_graph'],
                                meta_graph.name + '.metatxt')
        with open(filepath, 'w') as f:
            f.write(str(meta_graph))
        logger.info('Export meta graph into: {}'.format(filepath))


def Snapshot(tensors, filename, prefix='', suffix='.bin', format='default'):
    """Snapshot tensors into a binary file.

    Parameters
    ----------
    tensors : list of Tensor or Tensor
        The tensors to be wrote.
    filename : str
        The name of this binary file.
    prefix : str
        The prefix of this binary file.
    suffix : str
        The suffix of this binary file.
    format : str
        The format of this binary file.

    Returns
    -------
    None

    Notes
    -----
    The full filepath will be:  ``prefix`` + ``filename`` + ``suffix``.

    Available formats: ['default', 'caffe'].

    """
    from dragon.config import logger
    filepath = prefix + filename + suffix
    if mpi.Is_Init():
        if not mpi.AllowSnapshot(): return
        filepath = filepath + '.rank.{}'.format(mpi.Rank())

    dir = os.path.split(filepath)[0]
    if len(dir) > 0 and not os.path.exists(dir): os.makedirs(dir)

    if format == 'default':
        content = {}
        for tensor in tensors:
            content[tensor.name] = FetchTensor(tensor)
        with open(filepath, 'wb') as f:
            cPickle.dump(content, f, cPickle.HIGHEST_PROTOCOL)
        logger.info('Snapshot Model@: ' + filepath)
        logger.info('Model Format: cPickle')

    elif format is 'caffe':
        names = [tensor.name for tensor in tensors]
        SnapshotCC(filepath, names, 1)

    else: raise TypeError('Unknown binary format: {}'.format(format))


def Restore(filepath, format='default'):
    """Restore tensors from a binary file.

    Parameters
    ----------
    filepath : str
        The path of binary file.
    format : str
        The format of this binary file.

    Returns
    -------
    None

    Notes
    -----
    Available formats: ['default', 'caffe'].

    """
    from dragon.config import logger
    assert os.path.exists(filepath), 'model of path({}) does not exist.'.format(filepath)
    if format == 'default':
        content = cPickle.load(open(filepath, 'rb'))
        logger.info('Restore From Model@: ' + filepath)
        logger.info('Model Format: cPickle')
        for key, ndarray in content.items():
            if not HasTensor(key):
                logger.info('[Warning]:  Tensor({}) of model does not exist in any Graphs, skip.'.format(key))
            else:
                logger.info('[Info]: Tensor({}) restored.'.format(key))
                FeedTensor(key, ndarray)

    elif format == 'caffe':
        # TODO(PhyscalX): caffemodel can't save the tensor name
        # TODO(PhyscalX): we simply use layer_name + @paramX
        RestoreCC(filepath, 1)

    else:
        raise TypeError('Unknown binary format: {}'.format(format))