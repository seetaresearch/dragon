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

"""A Wrapper for the C++ backend Workspace.

Note that a default workspace is switched globally,
so these C++ calls are safe and deterministic.

See the documentation to learn how to switch between workspaces:

    <http://dragon.seetatech.com/api/python/contents/core/workspace.html>

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import threading
import six.moves.cPickle as pickle

import dragon.import_c_api as _C
import dragon.core.logging as logging
import dragon.proto.dragon_pb2 as pb

from dragon.config import GetGlobalOptions
from dragon.core import mpi, mapping, proto_utils


def CurrentWorkspace():
    """Return the current active workspace.

    Returns
    -------
    str
        The workspace name.

    """
    return _C.CurrentWorkspace()


def SwitchWorkspace(workspace_name, create_if_missing=True):
    """Switch to the specific workspace.

    Parameters
    ----------
    workspace_name : str
        The name of the specific workspace.
    create_if_missing : boolean
        Whether to create the specific workspace if it does not exist.

    Returns
    -------
    None

    """
    if workspace_name == '':
        raise ValueError('The workspace name should not be empty.')
    _C.SwitchWorkspace(workspace_name, create_if_missing)


def MoveWorkspace(target_ws, source_ws):
    """Move the source workspace into the target workspace.

    Parameters
    ----------
    target_ws : str
        The name of the target workspace.
    source_ws : str
        The name of the source workspace.

    Returns
    -------
    None

    """
    if target_ws == '' or source_ws == '':
        raise ValueError('The target or source name can not be empty.')
    _C.MoveWorkspace(target_ws, source_ws)


def ResetWorkspace(workspace_name=''):
    """Reset the specific workspace.

    Remove all resources of given workspace.

    If workspace name is empty, the current workspace will be modified.

    Parameters
    ----------
    workspace_name : str
        The name of the specific workspace.

    Returns
    -------
    None

    """
    _C.ResetWorkspace(workspace_name)


def ClearWorkspace(workspace_name=''):
    """Clear the specific workspace.

    You may need to clear the workspace when sharing grads.

    If workspace name is empty, the current workspace will be modified.

    Parameters
    ----------
    workspace_name : str
        The name of the specific workspace.

    Returns
    -------
    None

    """
    _C.ClearWorkspace(workspace_name)


def CreateGraph(graph_def):
    """Create the graph in the VM backend.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of meta graph.

    Returns
    -------
    str
        The graph name to run.

    """
    option = GetGlobalOptions()
    LogMetaGraph(graph_def)
    ExportMetaGraph(graph_def)
    return _C.CreateGraph(
        _stringify_proto(graph_def),
            option['log_optimized_graph'],
    )


def RunOperator(op_def, verbose=False):
    """Run the operator in the VM backend.

    Parameters
    ----------
    op_def : OperatorDef
        The definition of operator.
    verbose : boolean
        Whether to print the definition.

    Returns
    -------
    None

    """
    if isinstance(op_def, pb.OperatorDef):
        op_def = op_def.SerializeToString()
    _C.RunOperator(op_def, verbose)


def HasTensor(tensor):
    """Query whether tensor has registered in current workspace.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to query.

    Returns
    -------
    boolean
        The query result.

    """
    return _C.HasTensor(_stringify_tensor(tensor))


def CreateTensor(tensor):
    """Create the tensor in the backend.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to create.

    Returns
    -------
    None

    """
    return _C.CreateTensor(_stringify_tensor(tensor))


def CreateFiller(filler_def):
    """Create the filler in the backend.

    Parameters
    ----------
    filler_def : TensorFiller
        The filler.

    Returns
    -------
    None

    See Also
    --------
    `Tensor.Fill(*args, **kwargs)
    <tensor.html#dragon.core.tensor.Tensor.Fill>`_ - How to fill a Tensor. [**Caffe Style**]

    """
    filler_def = filler_def if isinstance(filler_def, str) \
        else filler_def.SerializePartialToString()
    _C.CreateFiller(filler_def)


def GetFillerType(tensor):
    """Get the filler type of specific tensor.

    It is useful if you want to tag some tensors,

    e.g. tag with ``numpy``, and get to initialize them lazily.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to query.

    Returns
    -------
    str
        The filler type.

    """
    return _C.GetFillerType(_stringify_tensor(tensor))


def GetTensorName(tensor):
    """Query the name represented in current workspace.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to query.

    Returns
    -------
    str
        The real(inplace-optimized) name in the backend.

    Notes
    -----
    The query result may be different from the one used in the frontend.

    """
    return _C.GetTensorName(_stringify_tensor(tensor))


def SetTensorAlias(tensor, alias):
    """Bind a alias to a existed tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to bind the alias.
    alias : str
        The alias.

    Returns
    -------
    None

    """
    return _C.SetTensorAlias(_stringify_tensor(tensor), alias)


def FetchTensor(tensor):
    """Fetch the values of given tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to fetch.

    Returns
    -------
    numpy.ndarray
        The values copied from the backend.

    """
    return _C.FetchTensor(_stringify_tensor(tensor))


def FeedTensor(tensor, array, force_cpu=False, dtype=None):
    """Feed the values to the given tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to feed.
    array : number, list, tuple, or numpy.ndarray
        The values to feed.
    force_cpu : boolean
        Whether force to feed to cpu context.
    dtype : str
        The data type. If ``None``, ``float32`` will be used instead.

    Returns
    -------
    None

    Examples
    --------
    >>> import dragon
    >>> a = dragon.Tensor().Variable()
    >>> dragon.workspace.FeedTensor(a, 1)
    >>> a_value = dragon.workspace.FetchTensor(a)
    >>> a_value, a_value.dtype
    >>> [ 1.], "float32"

    >>> dragon.workspace.FeedTensor(a, [[1, 2, 3]], dtype='float16')
    >>> a_value = a.get_value()
    >>> a_value, a_value.dtype
    >>> [[ 1.  2.  3.]], "float16"

    """
    name = tensor.name if hasattr(tensor, 'name') else str(tensor)
    if force_cpu is True:
        dev = proto_utils.GetDeviceOption('cpu')
    else:
        dev = proto_utils.GetDefaultDeviceOption()
        if dev is None: dev = proto_utils.GetGlobalDeviceOption()

    if not isinstance(array, numpy.ndarray):
        auto_data_type = numpy.float32 if dtype is None else dtype
    else:
        auto_data_type = array.dtype if dtype is None else dtype

    if hasattr(tensor, 'dtype') and tensor.dtype is not None:
        if tensor.dtype not in mapping.TENSOR_TYPE_TO_NP_TYPE:
            raise TypeError('Unsupported data type: {}'.format(tensor.dtype))
        preset_data_type = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.dtype]
        if dtype is not None:
            if dtype != preset_data_type:
                raise TypeError(
                    'The preset data type is {}, but force to {}'.
                        format(preset_data_type, dtype))
        auto_data_type = preset_data_type

    nd_array = numpy.array(array, dtype=auto_data_type, copy=False)
    _C.FeedTensor(name, nd_array, _stringify_proto(dev))


def ResetTensor(tensor):
    """Reset the memory of given tensor.

    Note that the tensor will not be ``DELETE`` for the workspace.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to reset.

    Returns
    -------
    None

    """
    return _C.ResetTensor(_stringify_tensor(tensor))


def RunGraph(
    graph_name, inputs=(), outputs=[],
        stage=None, return_outputs=True):
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
    None, NDArray or list of NDArray
        The outputs, format as NDArray.

    See Also
    --------
    `theano.function(*args, **kwargs)`_ - How to make a graph. [**Theano Style**]

    """
    # Explicit Feeding
    if len(inputs) > 0 and len(inputs[0]) > 0:
        if len(inputs[0]) != len(inputs[1]):
            raise RuntimeError(
                'Defined {} args, but {} are given.'
                    .format(len(inputs[0]), len(inputs[1])))
        for idx in range(len(inputs[0])):
            FeedTensor(inputs[0][idx], inputs[1][idx])

    # Run the graph according to the specified include/exclude rule
    runtime_stage = stage if stage else 'default'
    rule = _PREDEFINED_GRAPH_RUNTIME_STAGES[runtime_stage]
    _C.RunGraph(str(graph_name), str(rule['include']), str(rule['exclude']))

    # Try to return the outputs
    # Force to return may lead to asserts if outputs are not computed
    if return_outputs:
        if len(outputs) == 0 : return None
        elif len(outputs) == 1:  return outputs[0].get_value()
        else: return [outputs[i].get_value() for i in range(len(outputs))]


def FlowGradients(inputs, targets, input_grads=None, ignored_grads=None):
    """Compute the gradients of given input flows.

    Parameters
    ----------
    input_flow : sequence of OperatorDef
        The referring flows to generate gradient flows.
    targets : sequence or str
        The solving targets, generate grads automatically.
    input_grads : sequence of str or None
        The input grads.
    ignored_grads : sequence of str or None
        The grads that are explicitly ignored.

    Returns
    -------
    None

    """
    option = GetGlobalOptions()

    required_logging = True \
        if (option['log_optimized_graph'] or
            option['log_meta_graph']) else False

    _C.FlowGradients(
        inputs, targets,
            input_grads if input_grads else [],
                ignored_grads if ignored_grads else [],
                    option['share_grads'], required_logging)


def LogMetaGraph(graph_def):
    """Log the meta graph.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    """
    option = GetGlobalOptions()
    if option['log_meta_graph']: print(graph_def)


def ExportMetaGraph(graph_def):
    """Export the meta graph into a file under specific folder.

    You can set the exporting prefix by `config.ExportMetaGraph(prefix)`_.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of meta graph.

    Returns
    -------
    None

    """
    option = GetGlobalOptions()
    if option['export_meta_graph']:
        if not os.path.exists(option['export_meta_graph']):
            try:
                os.makedirs(option['export_meta_graph'])
            except Exception:
                raise ValueError('The given prefix is invalid.')

        path = os.path.join(
            option['export_meta_graph'],
                graph_def.name + '.metatxt')

        with open(path, 'w') as f: f.write(str(graph_def))
        logging.info('Export meta graph into: {}'.format(path))


def Snapshot(
    tensors, filename,
        prefix='', suffix='.bin',
            format='default'):
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
    The full file path will be:  ``prefix`` + ``filename`` + ``suffix``.

    Available formats: ['default', 'caffe'].

    """
    file_path = prefix + filename + suffix
    if mpi.Is_Init():
        if not mpi.AllowSnapshot(): return
        file_path = file_path + '.rank.{}'.format(mpi.Rank())

    dir = os.path.split(file_path)[0]
    if len(dir) > 0 and not os.path.exists(dir): os.makedirs(dir)

    if format == 'default':
        state_dict = {}
        for tensor in tensors:
            state_dict[tensor.name] = FetchTensor(tensor)
        with open(file_path, 'wb') as f:
            pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
        logging.info('Snapshot Model@: ' + file_path)
        logging.info('Model Format: Pickle')
    elif format is 'caffe':
        names = [tensor.name for tensor in tensors]
        _C.Snapshot(file_path, names, 1)
    else: raise TypeError('Unknown binary format: {}'.format(format))


def Restore(binary_file, format='default'):
    """Restore tensors from a binary file.

    Parameters
    ----------
    binary_file : str
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
    assert os.path.exists(binary_file), \
        'Binary file({}) does not exist.'.format(binary_file)

    if format == 'default':
        try:
            state_dict = pickle.load(open(binary_file, 'rb'))
        except UnicodeDecodeError:
            state_dict = pickle.load(open(binary_file, 'rb'), encoding='iso-8859-1')
        logging.info('Restore From Model@: ' + binary_file)
        logging.info('Model Format: Pickle')
        for k, v in state_dict.items():
            if HasTensor(k):
                FeedTensor(k, v)
                logging.info('[Info]: Tensor({}) is restored.'.format(k))
    elif format == 'caffe':
        # Caffe models can't save the tensor name
        # We simply use "layer_name/param:X"
        _C.Restore(binary_file, 1)
    else:
        raise TypeError('Unknown binary format: {}'.format(format))


def GetDummyName(basename, suffix='', domain='', zero_based=True):
    """Return a unique dummy name in current active workspace.

    The dummy name will be formatted ``basename`` + ``suffix``,
    or ``basename`` + ``_unique_index`` + ``suffix``.

    Names in the different ``domain`` could be same.

    Parameters
    ----------
    basename : str
        The basename.
    suffix : str
        The optional suffix adding to basename.
    domain : str
        The optional domain name.
    zero_based : boolean
        Whether number the name from 0.

    Returns
    -------
    str
        The unique dummy name.

    """
    return _C.GetDummyName(basename, suffix, domain, zero_based)


def _stringify_proto(obj):
    """Try to stringify a proto-buffer structure."""
    return obj.SerializeToString()


def _stringify_tensor(obj):
    """Try to stringify a tensor."""
    if hasattr(obj, 'name'): return str(obj.name)
    else: return str(obj)


# Define a global lock to lock the current workspace
_GLOBAL_WORKSPACE_LOCK = threading.Lock()

# Define some useful runtime stages
_PREDEFINED_GRAPH_RUNTIME_STAGES = {
    'default': {'include': '', 'exclude': ''},
    'forward': {'include': '', 'exclude': 'Gradient'},
    'backward': {'include': 'Gradient', 'exclude': 'Generate'},
    'backward_v2': {'include': 'Gradient', 'exclude': ''},
    'external_grads': {'include': '', 'exclude': 'Generate'},
}