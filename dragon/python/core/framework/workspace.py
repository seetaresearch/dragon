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

"""Wrappers for the Workspace of C++ backend.

Flexible API is provided to manage the global resources
between the Python threads (quite different from C++).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os

import numpy

from dragon import backend
from dragon.core.framework import config
from dragon.core.framework import mapping
from dragon.core.framework import proto_util
from dragon.core.framework import types
from dragon.core.proto import dragon_pb2
from dragon.core.util import logging
from dragon.core.util import tls
from dragon.core.util import six


class OperatorCollector(object):
    """A FIFO free list to manage the resource handle of operators.

    Operator who takes gradient will hold a handle,
    and it will be collected after the backward pass.

    Handles are collected according to the type,
    as the size of resources varies greatly.

    """

    def __init__(self):
        self._type2keys = collections.defaultdict(collections.deque)

    def alloc(self, op_type):
        """Allocate an unique handle according to type."""
        try:
            return self._type2keys[op_type].popleft()
        except IndexError:
            self._type2keys[op_type].append(
                get_dummy_name(
                    basename=op_type,
                    domain='Operator',
                    zero_based=False,
                ))
            return self._type2keys[op_type].popleft()

    def collect(self, handle):
        """Collect a unique handle."""
        op_type, _ = handle.split('_')
        self._type2keys[op_type].append(handle)


class TensorCollector(object):
    """A FIFO free list to manage the reused tensors.

    Tensors with the same scope are reused by turns,
    and thus, memory fragments will be reduced.

    Note that the fragments are inevitable due to the
    naive FIFO policy. Reset the workspace if the number
    of fragments is going to increase linearly.

    """

    def __init__(self):
        self._scope2keys = collections.defaultdict(collections.deque)

    def alloc(self, scope='${DATA}'):
        """Allocate an unique name under the specified scope."""
        try:
            return self._scope2keys[scope].popleft()
        except IndexError:
            self._scope2keys[scope].append(
                get_dummy_name(
                    basename='%s/Tensor' % scope,
                    domain='Tensor',
                    zero_based=False,
                ))
            return self._scope2keys[scope].popleft()

    def collect(self, name):
        """Collect a unique name."""
        if name.startswith('${'):
            scope, _ = name.split('/')
            self._scope2keys[scope].append(name)
            return True
        else:
            return False


class Workspace(backend.Workspace):
    """Space to isolate computations that share resources."""

    class Collectors(object):
        def __init__(self):
            self.TENSOR = TensorCollector()
            self.OPERATOR = OperatorCollector()

    def __init__(self, name=''):
        """Create a Workspace.

        Parameters
        ----------
        name : str, optional, default=''
            The optional workspace name.

        """
        super(Workspace, self).__init__(name)
        self._ref_objects = []
        self._collectors = self.Collectors()

    @property
    def collectors(self):
        """Return the resource collectors."""
        return self._collectors

    def merge_from(self, other):
        """Merge a external workspace into ``self``.

        The ``other`` will not be reset until ``self`` is reset.
        Carefulness should be taken to associate with the workspaces.

        Parameters
        ----------
        other : dragon.Workspace
            The given external workspace.

        Returns
        -------
        dragon.Workspace
            The ``self``.

        """
        self.MergeFrom(other)
        self._ref_objects.append(other)
        return self

    def as_default(self):
        """Switch ``self`` as the default workspace.

        Call this method with the **with** keyword.

        Once **with** is exited, the previous default will be set.

        Returns
        -------
        dragon.Workspace
            The ``self``.

        """
        return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_controller(self)

    def clear(self):
        """Remove all the tensors.

        Optionally call this method to clean the memories.

        """
        self.Clear()


def create_filler(filler_def):
    """Create a tensor filler in current workspace.

    Parameters
    ----------
    filler_def : TensorFiller
        The def of filler.

    """
    filler_def = filler_def if isinstance(filler_def, str) \
        else filler_def.SerializePartialToString()
    get_workspace().CreateFiller(filler_def)


def create_graph(graph_def):
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
    cfg = config.config()
    if cfg.graph_verbosity == 2:
        log_dir = cfg.log_dir
        if log_dir is not None:
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                except Exception:
                    raise ValueError('The given prefix is invalid.')
            path = os.path.join(
                log_dir,
                graph_def.name + '.txt',
            )
            with open(path, 'w') as f:
                f.write(str(graph_def))
            logging.info('Export meta graph to: %s' % path)
        else:
            print(graph_def)
    return get_workspace().CreateGraph(
        _stringify_proto(graph_def), cfg.graph_verbosity == 1)


def create_tensor(tensor):
    """Create the tensor in current workspace.

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to create.

    """
    tensor = _stringify_tensor(tensor)
    get_workspace().CreateTensor(tensor)


def feed_tensor(tensor, value, dtype=None, enforce_cpu=False):
    """Copy the value to tensor.

    Examples:

    ```python
    # Define a variable, feed then fetch the value
    x = dragon.Tensor().variable()
    dragon.workspace.feed_tensor(x, 1)
    print(dragon.workspace.fetch_tensor(x))

    # Feed by specifying a optional data type
    # Fetch through ``Tensor.get_value(...)``
    dragon.workspace.feed_tensor(a, [[1, 2, 3]], dtype='float16')
    print(x.get_value())
    ```

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to feed.
    value : array_like
        The value to copy.
    dtype : str, optional
        The optional data type.
    enforce_cpu : bool, optional, default=False
        **True** to copy using cpu context.

    """
    name = tensor.name if hasattr(tensor, 'name') else str(tensor)
    if enforce_cpu is True:
        dev = proto_util.get_device_option('cpu')
    else:
        dev = proto_util.get_default_device_option()
        if dev is None:
            dev = proto_util.get_global_device_option()

    # Steal the value from tensor storage if necessary.
    if types.is_tensor(value):
        value = getattr(value, 'get_value')()

    if not isinstance(value, numpy.ndarray):
        dtype = 'float32' if dtype is None else dtype
    else:
        dtype = value.dtype if dtype is None else dtype

    if hasattr(tensor, 'dtype') and tensor.dtype is not None:
        if tensor.dtype not in mapping.TENSOR_TYPE_TO_NP_TYPE:
            raise TypeError('Unsupported data type: %s' % tensor.dtype)
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.dtype]

    dev = _stringify_proto(dev)
    value = numpy.array(value, dtype=dtype, copy=False)
    get_workspace().FeedTensor(name, value, dev)


def fetch_tensor(tensor):
    """Return the value of tensor.

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to fetch.

    Returns
    -------
    numpy.ndarray
        The array copied from backend.

    """
    tensor = _stringify_tensor(tensor)
    return get_workspace().FetchTensor(tensor)


def get_dummy_name(basename, suffix='', domain='', zero_based=True):
    """Return an unique dummy name in current workspace.

    The dummy name will be formatted as:
    <basename> + <unique_index> + <suffix>.

    Names in the different ``domain`` could be same.

    Parameters
    ----------
    basename : str
        The basename.
    suffix : str, optional
        The optional suffix adding to basename.
    domain : str, optional
        The optional domain name.
    zero_based : bool, optional, default=True
        Whether number the index from 0.

    Returns
    -------
    str
        The unique dummy name.

    """
    return get_workspace().GetDummyName(
        basename, suffix, domain, zero_based)


def get_tensor_name(tensor):
    """Return the name of tensor in current workspace.

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to query.

    Returns
    -------
    str
        The tensor name.

    """
    tensor = _stringify_tensor(tensor)
    return get_workspace().GetTensorName(tensor)


def get_workspace():
    """Return the current default workspace.

    Returns
    -------
    dragon.Workspace
        The default workspace.

    """
    return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_default()


def has_tensor(tensor):
    """Return a bool indicating if tensor is in current workspace.

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to query.

    Returns
    -------
    bool
        **True** if specified tensor is existing otherwise **False**.

    """
    tensor = _stringify_tensor(tensor)
    return get_workspace().HasTensor(tensor)


def load(file_path, format='pkl'):
    """Load tensors from a binary file.

    Parameters
    ----------
    file_path : str
        The path of binary file.
    format : {'pkl', 'caffe'}, optional
        The serializing format.

    """
    assert os.path.exists(file_path), \
        'File(%s) does not exist.' % file_path
    if format == 'pkl':
        try:
            with open(file_path, 'rb') as f:
                state_dict = six.moves.pickle.load(f)
        except UnicodeDecodeError:
            with open(file_path, 'rb') as f:
                state_dict = six.moves.pickle.load(f, encoding='iso-8859-1')
        logging.info('Load From Model@: ' + file_path)
        logging.info('Model Format: Pickle')
        for k, v in state_dict.items():
            if has_tensor(k):
                feed_tensor(k, v)
                logging.info('Tensor({}) is loaded.'.format(k))
    elif format == 'caffe':
        get_workspace().Load(file_path, 1)
    else:
        raise TypeError('Unknown binary format: ' + format)


def reset_tensor(tensor):
    """Reset the memory of tensor.

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to reset.

    """
    tensor = _stringify_tensor(tensor)
    return get_workspace().ResetTensor(tensor)


def reset_workspace():
    """Reset the current default workspace."""
    if not _GLOBAL_DEFAULT_WORKSPACE_STACK.is_cleared():
        raise AssertionError(
            "Do not use reset_default() to clear "
            "nested workspaces.\nIf you need a cleared workspace, "
            "exit the nesting and create a new workspace.")
    _GLOBAL_DEFAULT_WORKSPACE_STACK.reset()


def run_backward(
    forward_ops,
    targets,
    sources=None,
    input_grads=None,
    ignored_grads=None,
):
    """Compute the gradients of input operators.

    Parameters
    ----------
    forward_ops : Sequence[OperatorDef]
        The referring operators to generate gradients.
    targets : Sequence[str]
        The solving targets.
    sources : Sequence[str], optional
        The optional sources to hook the intermediate grads.
    input_grads : Sequence[str], optional
        The external input grads.
    ignored_grads : Sequence[str], optional
        The grads that are explicitly ignored.

    """
    cfg = config.config()
    get_workspace().RunBackward(
        forward_ops,
        targets,
        sources if sources else [],
        input_grads if input_grads else [],
        ignored_grads if ignored_grads else [],
        cfg.graph_optimization > 2,
        cfg.graph_verbosity > 0,
    )


def run_graph(
    graph,
    inputs=(),
    outputs=(),
    stage=None,
    return_outputs=True,
):
    """Run the graph in current workspace.

    Parameters
    ----------
    graph : str
        The name of graph.
    inputs : tuple
        The **inputs** and **values**.
    outputs : Sequence[dragon.Tensor]
        The outputs of the graph.
    stage : str, optional
        The preset custom stages.
    return_outputs : bool, optional, default=False
        Whether to return the outputs.

    Returns
    -------
    Sequence[numpy.ndarray]
        The outputs which are copied to numpy array.

    """
    # The explicit feeding.
    if len(inputs) > 0 and len(inputs[0]) > 0:
        if len(inputs[0]) != len(inputs[1]):
            raise RuntimeError(
                'Defined {} args, but {} are given.'
                .format(len(inputs[0]), len(inputs[1]))
            )
        for idx in range(len(inputs[0])):
            feed_tensor(inputs[0][idx], inputs[1][idx])

    # Run the graph according to the specified include/exclude rule.
    runtime_stage = stage if stage else 'default'
    rule = _PREDEFINED_GRAPH_RUNTIME_STAGES[runtime_stage]
    get_workspace().RunGraph(
        graph, rule['include'], rule['exclude'])

    # Try to return the outputs.
    # Force to return may lead to asserts if outputs are not computed.
    if return_outputs:
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0].get_value()
        else:
            return [outputs[i].get_value() for i in range(len(outputs))]


def run_operator(op_def):
    """Run the operator(s) in current workspace.

    Parameters
    ----------
    op_def : Union[OperatorDef, Sequence[OperatorDef]]
        The definition of operator(s).

    """
    cfg = config.config()
    if isinstance(op_def, dragon_pb2.OperatorDef):
        op_def = op_def.SerializeToString()
    get_workspace().RunOperator(op_def, cfg.graph_verbosity > 0)


def save(
    tensors,
    filename,
    prefix='',
    suffix='.pkl',
    format='pkl',
):
    """Serialize tensors into a binary file.

    The file path is formatted as:
    <prefix> + <filename> + <suffix>

    Parameters
    ----------
    tensors : Sequence[dragon.Tensor]
        The tensors to be wrote.
    filename : str
        The filename.
    prefix : str, optional, default=''
        The prefix.
    suffix : str, optional, default='.pkl'
        The suffix.
    format : {'pkl', 'caffe'}, optional
        The serializing format.

    """
    file_path = prefix + filename + suffix
    dir = os.path.split(file_path)[0]
    if len(dir) > 0 and not os.path.exists(dir):
        os.makedirs(dir)
    if format == 'pkl':
        state_dict = {}
        for tensor in tensors:
            state_dict[tensor.name] = fetch_tensor(tensor)
        with open(file_path, 'wb') as f:
            six.moves.pickle.dump(
                state_dict, f,
                six.moves.pickle.HIGHEST_PROTOCOL,
            )
        logging.info('Save model to: ' + file_path)
        logging.info('Model Format: Pickle')
    elif format == 'caffe':
        names = [tensor.name for tensor in tensors]
        get_workspace().Save(file_path, names, 1)
    else:
        raise TypeError('Unknown binary format: ' + format)


def set_tensor_alias(tensor, alias):
    """Bind an alias to an existing tensor.

    Parameters
    ----------
    tensor : Union[dragon.Tensor, str]
        The tensor to bind the alias.
    alias : str
        The alias.

    """
    tensor = _stringify_tensor(tensor)
    get_workspace().SetTensorAlias(tensor, alias)


def _stringify_proto(obj):
    """Try to stringify a proto-buffer structure."""
    return obj.SerializeToString()


def _stringify_tensor(obj):
    """Try to stringify a tensor."""
    if hasattr(obj, 'id'):
        return str(obj.id)
    else:
        return str(obj)


class _DefaultWorkspaceStack(tls.Stack):
    """A thread-local stack for default workspaces."""

    def __init__(self):
        super(_DefaultWorkspaceStack, self).__init__()
        self._global_default_workspace = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        ret = super(_DefaultWorkspaceStack, self).get_default()
        if ret is None:
            ret = self._get_default()
        return ret

    def _get_default(self):
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


# Define a global stack to store the workspaces of current thread.
_GLOBAL_DEFAULT_WORKSPACE_STACK = _DefaultWorkspaceStack()

# Define some useful runtime stages.
_PREDEFINED_GRAPH_RUNTIME_STAGES = {
    'default': {'include': '', 'exclude': ''},
    'forward': {'include': '', 'exclude': 'Gradient'},
    'backward': {'include': 'Gradient', 'exclude': 'Generate'},
    'backward_v2': {'include': 'Gradient', 'exclude': ''},
    'external_grads': {'include': '', 'exclude': 'Generate'},
}
