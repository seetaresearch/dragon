# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Generic interfaces of current default workspace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import numpy

from dragon import backend
from dragon.core.framework import config
from dragon.core.framework import mapping
from dragon.core.framework import proto_util
from dragon.core.framework import types
from dragon.core.proto import dragon_pb2
from dragon.core.util import serialization
from dragon.core.util import tls


class OpCollector(object):
    """A FIFO free list to manage the resource handle of operators."""

    def __init__(self, parent):
        self._parent = parent
        self._type2keys = collections.defaultdict(collections.deque)

    def alloc(self, op_type):
        """Allocate an unique handle according to type."""
        try:
            return self._type2keys[op_type].popleft()
        except IndexError:
            self._type2keys[op_type].append(
                self._parent.unique_name(
                    name=op_type,
                    namespace='Op',
                    zero_based=False))
            return self._type2keys[op_type].popleft()

    def collect(self, handle):
        """Collect an unique handle."""
        op_type, _ = handle.split('_')
        self._type2keys[op_type].append(handle)


class TensorCollector(object):
    """A FIFO free list to manage the reused tensors."""

    def __init__(self, parent):
        self._parent = parent
        self._scope2keys = collections.defaultdict(collections.deque)

    def alloc(self, scope='${DATA}'):
        """Allocate an unique name under the specified scope."""
        try:
            return self._scope2keys[scope].popleft()
        except IndexError:
            self._scope2keys[scope].append(
                self._parent.unique_name(
                    name='%s/Tensor' % scope,
                    namespace='Tensor',
                    zero_based=False))
            return self._scope2keys[scope].popleft()

    def collect(self, name):
        """Collect an unique name."""
        scope, _ = name.split('/')
        self._scope2keys[scope].append(name)


class Workspace(backend.Workspace):
    """Sandbox to isolate the resources and computations."""

    class Collectors(object):
        def __init__(self, workspace):
            self.OP = OpCollector(workspace)
            self.TENSOR = TensorCollector(workspace)

    def __init__(self, name=''):
        """Create a ``Workspace``.

        Parameters
        ----------
        name : str, optional, default=''
            The optional workspace name.

        """
        super(Workspace, self).__init__(name)
        self._references = []
        self._collectors = self.Collectors(self)

    @property
    def collectors(self):
        """Return the resource collectors."""
        return self._collectors

    def as_default(self):
        """Switch this workspace as the default.

        Call this method with the **with** keyword.

        Once **with** is exited, the previous default will be set.

        Returns
        -------
        dragon.Workspace
            This workspace.

        """
        return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_controller(self)

    def clear(self):
        """Clear the cached tensors, operators and graphs.

        Call this method before deleting to free cached resources certainly:

        ```python
        my_workspace = dragon.Workspace()
        my_workspace.clear()
        del my_workspace
        ```

        """
        self.Clear()

    def create_graph(self, graph_def):
        """Create the graph.

        Parameters
        ----------
        graph_def : GraphDef
            The ``GraphDef`` protocol buffer.

        Returns
        -------
        str
            The graph name.

        """
        cfg = config.config()
        if cfg.graph_verbosity == 2:
            print(graph_def)
        return self.CreateGraph(
            serialization.serialize_proto(graph_def),
            cfg.graph_verbosity == 1)

    def create_tensor(self, name, filler_info=None):
        """Create the tensor.

        Parameters
        ----------
        name : str
            The tensor name.
        filler_info : FillerInfo
            The ``FillerInfo`` protocol buffer.

        Returns
        -------
        TensorImpl
            The tensor implementation.

        """
        return self.CreateTensor(
            name, serialization.serialize_proto(filler_info))

    def feed_tensor(self, tensor, value, dtype=None, enforce_cpu=False):
        """Copy the value to tensor.

        Examples:

        ```python
        # Define a named tensor to feed
        x = dragon.Tensor(name='x')
        dragon.get_workspace().feed_tensor(x, 0)

        # Feed by specifying a tensor name
        # Note that it will create the implementation whatever
        dragon.get_workspace().feed_tensor('y', 1)
        print(dragon.get_workspace().has_tensor('y'))  # True
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
        if types.is_tensor(value):
            # Steal the data if value is a tensor
            value = getattr(value, 'get_value')()
        # Determine the data type from argument or value
        if not isinstance(value, numpy.ndarray):
            dtype = 'float32' if dtype is None else dtype
        else:
            dtype = value.dtype if dtype is None else dtype
        if hasattr(tensor, 'dtype') and tensor.dtype is not None:
            if tensor.dtype not in mapping.TENSOR_TYPE_TO_NP_TYPE:
                raise TypeError('Unsupported data type: ' + tensor.dtype)
            dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.dtype]
        # Determine the copying device option
        if enforce_cpu is True:
            device_option = proto_util.get_device_option('cpu')
        else:
            device_option = proto_util.get_default_device_option()
            if device_option is None:
                device_option = proto_util.get_global_device_option()
        # Copy data to the backend
        self.FeedTensor(
            _stringify_object(tensor),
            numpy.array(value, dtype=dtype, copy=False),
            serialization.serialize_proto(device_option),
        )

    def fetch_tensor(self, tensor):
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
        return self.FetchTensor(_stringify_object(tensor))

    def has_tensor(self, tensor):
        """Return whether the tensor is in this workspace.

        Parameters
        ----------
        tensor : Union[dragon.Tensor, str]
            The tensor.

        Returns
        -------
        bool
            **True** if tensor is existing otherwise **False**.

        """
        return self.HasTensor(_stringify_object(tensor))

    def merge_from(self, other):
        """Merge resources from the other.

        The ``other`` will not be reset until ``self`` is reset.
        Carefulness should be taken to associate with the workspaces.

        Parameters
        ----------
        other : dragon.Workspace
            The workspace to merge.

        Returns
        -------
        dragon.Workspace
            This workspace.

        """
        self.MergeFrom(other)
        self._references.append(other)
        return self

    def register_alias(self, target, alias):
        """Register an alias for the target.

        Parameters
        ----------
        target : Union[str, dragon.Tensor]
            The string or named object.
        alias : str
            The alias.

        """
        self.RegisterAlias(_stringify_object(target), alias)

    def reset_tensor(self, tensor):
        """Reset the tensor.

        Parameters
        ----------
        tensor : Union[dragon.Tensor, str]
            The tensor to reset.

        """
        self.ResetTensor(_stringify_object(tensor))

    def run_backward(
        self,
        op_defs,
        targets,
        sources=None,
        input_grads=None,
        empty_grads=None,
    ):
        """Compute the gradients of input operators.

        Parameters
        ----------
        op_defs : Sequence[OperatorDef]
            The executed op defs.
        targets : Sequence[str]
            The derivative targets.
        sources : Sequence[str], optional
            The differentiated inputs.
        input_grads : Sequence[str], optional
            The input grad for targets.
        empty_grads : Sequence[str], optional
            The grads to set to empty.

        """
        cfg = config.config()
        self.RunBackward(
            op_defs,
            targets,
            sources if sources else [],
            input_grads if input_grads else [],
            empty_grads if empty_grads else [],
            cfg.graph_optimization <= 2,
            cfg.graph_verbosity > 0,
        )

    def run_graph(
        self,
        name,
        inputs_and_values=None,
        outputs=None,
        executing_stage=None,
        return_outputs=True,
    ):
        """Run the graph.

        Parameters
        ----------
        name : str
            The graph name.
        inputs_and_values : Tuple[Sequence, Sequence], optional
            The input tensors and feeding values.
        outputs : Sequence[dragon.Tensor], optional
            The output tensors.
        executing_stage : str, optional
            The optional executing stage.
        return_outputs : bool, optional, default=False
            Whether to return the output values.

        """
        # The explicit feeding for inputs.
        if inputs_and_values is not None:
            inputs, values = inputs_and_values
            if len(inputs) != len(values):
                raise ValueError(
                    'Specified %d values for %d inputs.'
                    % (len(values), len(inputs)))
            for tensor, value in zip(inputs, values):
                self.feed_tensor(tensor, value)
        # Run the graph according to the specified include/exclude rule.
        stage_str = executing_stage if executing_stage else 'default'
        exec_stage = _PREDEFINED_GRAPH_EXECUTING_STAGES[stage_str]
        self.RunGraph(name, exec_stage['include'], exec_stage['exclude'])
        # Maybe return the output values.
        if return_outputs and outputs is not None:
            if len(outputs) == 1:
                return outputs[0].get_value()
            else:
                return [outputs[i].get_value() for i in range(len(outputs))]

    def run_operator(self, op_def):
        """Run the operator.

        Parameters
        ----------
        op_def : Union[OperatorDef, Sequence[OperatorDef]]
            The ``OperatorDef`` protocol buffer.

        """
        cfg = config.config()
        if isinstance(op_def, dragon_pb2.OperatorDef):
            op_def = op_def.SerializePartialToString()
        self.RunOperator(op_def, cfg.graph_verbosity > 0)

    def unique_name(self, name, suffix='', namespace='', zero_based=True):
        """Return an unique name.

        Names in the different ``namespace`` could be same.

        Parameters
        ----------
        name : str
            The name to make unique.
        suffix : str, optional
            The optional suffix adding to name.
        namespace : str, optional
            The optional scope to make unique within.
        zero_based : bool, optional, default=True
            **True** to number the index from 0 otherwise 1.

        Returns
        -------
        str
            The unique name.

        """
        return self.UniqueName(name, suffix, namespace, zero_based)


def get_workspace():
    """Return the current default workspace.

    Returns
    -------
    dragon.Workspace
        The default workspace.

    """
    return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_default()


def reset_workspace():
    """Reset the current default workspace."""
    if not _GLOBAL_DEFAULT_WORKSPACE_STACK.is_cleared():
        raise AssertionError(
            "Do not use reset_workspace() to clear "
            "nested workspaces.\nIf you need a cleared workspace, "
            "exit the nesting and create a new workspace.")
    _GLOBAL_DEFAULT_WORKSPACE_STACK.reset()


def _stringify_object(obj):
    """Try to stringify a object."""
    return obj.id if hasattr(obj, 'id') else obj


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
        if self._global_default_workspace is not None:
            self._global_default_workspace.clear()
        self._global_default_workspace = None

    @contextlib.contextmanager
    def get_controller(self, default):
        with super(_DefaultWorkspaceStack, self) \
                .get_controller(default) as g:
            yield g


# Global stack to store the workspaces of current thread.
_GLOBAL_DEFAULT_WORKSPACE_STACK = _DefaultWorkspaceStack()

# Predefined graph executing stages.
_PREDEFINED_GRAPH_EXECUTING_STAGES = {
    'default': {'include': '', 'exclude': ''},
    'forward': {'include': '', 'exclude': '.*Gradient.*'},
    'backward': {'include': '.*Gradient.*', 'exclude': 'GradientGenerate'},
    'backward_v2': {'include': '.*Gradient.*', 'exclude': ''},
}
