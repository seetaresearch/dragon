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

"""Wrappers for the Workspace of C++ backend.

Flexible API is provided to manage the global resources
between the Python threads (quite different from C++).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy
import contextlib
import six.moves.cPickle as pickle
from collections import defaultdict, deque

from dragon import config as _cfg
from dragon import import_c_api as _C
from dragon.core import tls as _tls
from dragon.core import mpi as _mpi
from dragon.core import logging as _logging
from dragon.core import mapping as _mapping
from dragon.proto import dragon_pb2 as _proto_def
from dragon.core import proto_utils as _proto_utils


class TensorPool(object):
    """We apply the TensorPool to manage the reused tensors.

    Tensors with the same scope in the pool will be reused by turns,
    which speeds up the whole system by reducing the unnecessary deconstructing.

    Heuristically, we have used 5 pools with different scopes:

    * scope(Leaf): A Pool to reuse leaf tensors.

    * scope(NumPy): A pool to reuse leaf tensors from numpy.

    * scope(Join): A pool to reuse RT(runtime) tensors required by forward-backward.

    * scope(Detach): A pool to reuse RT(runtime) tensors required by forward only.

    * scope(Reference): A pool to reuse reshaped tensors(sharing contents).

    """
    def __init__(self):
        # deque provide much higher performance than Queue
        self._scope2keys = defaultdict(deque)

    def get(self, scope='${DETACH}'):
        try:
            return self._scope2keys[scope].popleft()
        except:
            self._scope2keys[scope].append(
                GetDummyName(
                    '${POOL}/%s/Tensor' % scope,
                        domain='Tensor', zero_based=False))
            return self._scope2keys[scope].popleft()

    def put(self, name):
        if '${POOL}' in name:
            scope, _ = name[8:].split('/')
            self._scope2keys[scope].append(name)
            return True
        else: return False


class OperatorPool(object):
    """Operators whose gradients is required will hold a resource handle,
    which is also called ``Anchor`` in the backend.

    We apply this pool to collect the handles according to the type of operator,
    as the mem size of temporal resources varies greatly.

    The resource handle will be released after the gradient flow automatically.

    """
    def __init__(self):
        # deque provide much higher performance than Queue
        self._type2keys = defaultdict(deque)

    def get(self, op_type):
        try:
            return self._type2keys[op_type].popleft()
        except:
            self._type2keys[op_type].append(
                GetDummyName(
                    '${POOL}/%s' % op_type,
                        domain='Operator', zero_based=False))
            return self._type2keys[op_type].popleft()

    def put(self, op_name):
        op_type, _ = op_name[8:].split('_')
        self._type2keys[op_type].append(op_name)


class Workspace(_C.Workspace):
    """A wrapper for the C implemented workspace.

    This class is a fusion of *Workspace*, *Pool* and *tf.Graph*.

    We find that they work in a similar way while named differently.

    """
    def __init__(self, name=''):
        super(Workspace, self).__init__(name)
        self._ref_objects = []
        self._collections = {}
        self.tensor_pool = TensorPool()
        self.operator_pool = OperatorPool()

    def get_collection_ref(self, name):
        coll_list = self._collections.get(name, None)
        if coll_list is None:
            coll_list = []
            self._collections[name] = coll_list
        return coll_list

    def get_collection(self, name, scope=None):
        coll_list = self._collections.get(name, None)
        if coll_list is None:
            return []
        if scope is None:
            return list(coll_list)
        else:
            filter_coll_list = []
            regex = re.compile(scope)
            for item in coll_list:
                if hasattr(item, "name") and regex.match(item.name):
                    filter_coll_list.append(item)
            return filter_coll_list

    def add_to_collection(self, name, value):
        if name not in self._collections:
            self._collections[name] = [value]
        else:
            self._collections[name].append(value)

    def add_to_collections(self, names, value):
        for name in names:
            self.add_to_collection(name, value)

    def merge_from(self, other):
        """Merge a external workspace into ``self``.

        The ``other`` will not be reset until ``self`` is reset.
        Carefulness should be taken to associate with the workspaces.

        Parameters
        ----------
        other : Workspace
            The given external workspace.

        Returns
        -------
        Workspace
            The ``self``.

        """
        self.MergeFrom(other)
        self._ref_objects.append(other)
        return self

    def as_default(self):
        """Switch ``self`` as the default workspace.

        Call this method with the *with* keyword.

        Once *with* is exited, the previous default will be set.

        Returns
        -------
        Workspace
            The ``self``.

        """
        return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_controller(self)

    def clear(self):
        """Remove all the tensors.

        Optionally call this method to clean the memories.

        Returns
        -------
        None

        """
        self.Clear()


def get_default_workspace():
    """Return the current default workspace.

    Returns
    -------
    Workspace
        The default workspace.

    """
    return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_default()


def reset_default_workspace():
    """Reset the global default workspace.

    Do not call this method to reset any instances.

    Returns
    -------
    None

    """
    if not _GLOBAL_DEFAULT_WORKSPACE_STACK.is_cleared():
        raise AssertionError(
            "Do not use reset_default_workspace() to clear "
            "nested workspaces.\nIf you need a cleared workspace, "
            "exit the nesting and create a new workspace.")
    _GLOBAL_DEFAULT_WORKSPACE_STACK.reset()


def CreateGraph(graph_def):
    """Create the graph in current workspace.

    Parameters
    ----------
    graph_def : GraphDef
        The definition of meta graph.

    Returns
    -------
    str
        The graph name to run.

    """
    LogMetaGraph(graph_def)
    ExportMetaGraph(graph_def)
    options = _cfg.GetGlobalOptions()
    return get_default_workspace().CreateGraph(
        _stringify_proto(graph_def),
            options['log_optimized_graph'])


def RunOperator(op_def, verbose=False):
    """Run the operator.

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
    if isinstance(op_def, _proto_def.OperatorDef):
        op_def = op_def.SerializeToString()
    get_default_workspace().RunOperator(op_def, verbose)


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
    tensor = _stringify_tensor(tensor)
    return get_default_workspace().HasTensor(tensor)


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
    tensor = _stringify_tensor(tensor)
    get_default_workspace().CreateTensor(tensor)


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
    get_default_workspace().CreateFiller(filler_def)


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
    tensor = _stringify_tensor(tensor)
    return get_default_workspace().GetFillerType(tensor)


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
    tensor = _stringify_tensor(tensor)
    return get_default_workspace().GetTensorName(tensor)


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
    tensor = _stringify_tensor(tensor)
    get_default_workspace().SetTensorAlias(tensor, alias)


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
    tensor = _stringify_tensor(tensor)
    return get_default_workspace().FetchTensor(tensor)


def FeedTensor(
    tensor,
    array,
    force_cpu=False,
    dtype=None,
):
    """Feed the values to the given tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to feed.
    array : array_like
        The values to feed.
    force_cpu : boolean, optional, default=False
        Whether force to feed to cpu context.
    dtype : str, optional
        The optional data type.

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
        dev = _proto_utils.GetDeviceOption('cpu')
    else:
        dev = _proto_utils.GetDefaultDeviceOption()
        if dev is None: dev = _proto_utils.GetGlobalDeviceOption()

    if not isinstance(array, numpy.ndarray):
        dtype = 'float32' if dtype is None else dtype
    else:
        dtype = array.dtype if dtype is None else dtype

    if hasattr(tensor, 'dtype') and tensor.dtype is not None:
        if tensor.dtype not in _mapping.TENSOR_TYPE_TO_NP_TYPE:
            raise TypeError('Unsupported data type: {}'.format(tensor.dtype))
        dtype = _mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.dtype]

    dev = _stringify_proto(dev)
    array = numpy.array(array, dtype=dtype, copy=False)
    get_default_workspace().FeedTensor(name, array, dev)


def ResetTensor(tensor):
    """Reset the memory of given tensor.

    Parameters
    ----------
    tensor : Tensor or str
        The tensor to reset.

    Returns
    -------
    None

    """
    tensor = _stringify_tensor(tensor)
    return get_default_workspace().ResetTensor(tensor)


def RunGraph(
    graph_name,
    inputs=(),
    outputs=[],
    stage=None,
    return_outputs=True,
):
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
    sequence of numpy.ndarray
        The outputs which are copied to numpy array.

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
    get_default_workspace().RunGraph(
        graph_name, rule['include'], rule['exclude'])

    # Try to return the outputs
    # Force to return may lead to asserts if outputs are not computed
    if return_outputs:
        if len(outputs) == 0 : return None
        elif len(outputs) == 1:  return outputs[0].get_value()
        else: return [outputs[i].get_value() for i in range(len(outputs))]


def Backward(
    forward_ops,
    targets,
    input_grads=None,
    ignored_grads=None,
):
    """Compute the gradients of given input operators.

    Parameters
    ----------
    forward_ops : sequence of OperatorDef
        The referring ops to generate gradients.
    targets : sequence or str
        The solving targets.
    input_grads : sequence of str, optional
        The external input grads.
    ignored_grads : sequence of str, optional
        The grads that are explicitly ignored.

    Returns
    -------
    None

    """
    options = _cfg.GetGlobalOptions()

    required_logging = True \
        if (options['log_optimized_graph'] or
            options['log_meta_graph']) else False

    get_default_workspace().Backward(
        forward_ops,
        targets,
        input_grads if input_grads else [],
        ignored_grads if ignored_grads else [],
        options['share_grads'],
        required_logging,
    )


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
    options = _cfg.GetGlobalOptions()
    if options['log_meta_graph']: print(graph_def)


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
    options = _cfg.GetGlobalOptions()
    if options['export_meta_graph']:
        if not os.path.exists(options['export_meta_graph']):
            try:
                os.makedirs(options['export_meta_graph'])
            except Exception:
                raise ValueError('The given prefix is invalid.')

        path = os.path.join(
            options['export_meta_graph'],
                graph_def.name + '.metatxt')

        with open(path, 'w') as f: f.write(str(graph_def))
        _logging.info('Export meta graph into: {}'.format(path))


def Snapshot(
    tensors,
    filename,
    prefix='',
    suffix='.bin',
    format='pickle',
):
    """Serialize tensors into a binary file.

    The filename is formatted as:

        ``prefix`` + ``filename`` + ``suffix``

    Parameters
    ----------
    tensors : list of Tensor or Tensor
        The tensors to be wrote.
    filename : str
        The name of this binary file.
    prefix : str, optional, default=''
        The prefix of this binary file.
    suffix : str, optional, default='.bin'
        The suffix of this binary file.
    format : {'pickle', 'caffe'}, optional
        The format of this binary file.

    Returns
    -------
    None

    Notes
    -----


    """
    file_path = prefix + filename + suffix

    if _mpi.Is_Init():
        if not _mpi.AllowSnapshot(): return
        file_path = file_path + '.rank.{}'.format(_mpi.Rank())

    dir = os.path.split(file_path)[0]
    if len(dir) > 0 and not os.path.exists(dir): os.makedirs(dir)

    if format == 'pickle':
        state_dict = {}
        for tensor in tensors:
            state_dict[tensor.name] = FetchTensor(tensor)
        with open(file_path, 'wb') as f:
            pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
        _logging.info('Snapshot Model@: ' + file_path)
        _logging.info('Model Format: Pickle')
    elif format == 'caffe':
        names = [tensor.name for tensor in tensors]
        get_default_workspace().Snapshot(file_path, names, 1)
    else:
        raise TypeError('Unknown binary format: ' + format)


def Restore(binary_file, format='pickle'):
    """Restore tensors from a binary file.

    Parameters
    ----------
    binary_file : str
        The path of binary file.
    format : {'pickle', 'caffe'}, optional
        The format of this binary file.

    Returns
    -------
    None

    """
    assert os.path.exists(binary_file), \
        'Binary file({}) does not exist.'.format(binary_file)

    if format == 'pickle':
        try:
            state_dict = pickle.load(open(binary_file, 'rb'))
        except UnicodeDecodeError:
            state_dict = pickle.load(
                open(binary_file, 'rb'), encoding='iso-8859-1')
        _logging.info('Restore From Model@: ' + binary_file)
        _logging.info('Model Format: Pickle')
        for k, v in state_dict.items():
            if HasTensor(k):
                FeedTensor(k, v)
                _logging.info('Tensor({}) is restored.'.format(k))
    elif format == 'caffe':
        get_default_workspace().Restore(binary_file, 1)
    else:
        raise TypeError('Unknown binary format: ' + format)


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
    return get_default_workspace().GetDummyName(
        basename, suffix, domain, zero_based)


def _stringify_proto(obj):
    """Try to stringify a proto-buffer structure."""
    return obj.SerializeToString()


def _stringify_tensor(obj):
    """Try to stringify a tensor."""
    if hasattr(obj, 'name'): return str(obj.name)
    else: return str(obj)


class _DefaultWorkspaceStack(_tls.Stack):
    """A thread-local stack of objects for
    providing an implicit default workspace.

    """
    def __init__(self):
        super(_DefaultWorkspaceStack, self).__init__()
        self._global_default_workspace = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        ret = super(_DefaultWorkspaceStack, self).get_default()
        if ret is None: ret = self._get_default_workspace()
        return ret

    def _get_default_workspace(self):
        if self._global_default_workspace is None:
            self._global_default_workspace = Workspace()
        return self._global_default_workspace

    def reset(self):
        super(_DefaultWorkspaceStack, self).reset()
        self._global_default_workspace = None

    @contextlib.contextmanager
    def get_controller(self, default):
        with super(_DefaultWorkspaceStack, self) \
                .get_controller(default) as g:
            yield g


# Define a global stack to store the workspaces of current thread
_GLOBAL_DEFAULT_WORKSPACE_STACK = _DefaultWorkspaceStack()

# Define some useful runtime stages
_PREDEFINED_GRAPH_RUNTIME_STAGES = {
    'default': {'include': '', 'exclude': ''},
    'forward': {'include': '', 'exclude': 'Gradient'},
    'backward': {'include': 'Gradient', 'exclude': 'Generate'},
    'backward_v2': {'include': 'Gradient', 'exclude': ''},
    'external_grads': {'include': '', 'exclude': 'Generate'},
}