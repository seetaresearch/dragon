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
"""Functions and helpers for workspace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import heapq
import weakref

from dragon.core.framework import backend
from dragon.core.framework import config
from dragon.core.proto import dragon_pb2
from dragon.core.util import logging
from dragon.core.util import serialization
from dragon.core.util import tls


class Workspace(object):
    """Standalone environment for resources and computations."""

    class HandlePool(object):
        """Pool to manage the handles."""

        def __init__(self, parent):
            self._weak_parent = weakref.ref(parent)
            self._handles = collections.defaultdict(list)

        def create(self, key):
            """Create a handle."""
            try:
                return key + '_' + str(heapq.heappop(self._handles[key]))
            except IndexError:
                return self._weak_parent().unique_name(
                    name=key, namespace='WorkspaceHandle', zero_based=False)

        def release(self, handle):
            """Release a created handle."""
            try:
                key, index = handle.rsplit('_', 1)
            except ValueError:
                return
            try:
                heapq.heappush(self._handles[key], int(index))
            except AttributeError:
                pass

    def __init__(self):
        """Create a ``Workspace``."""
        self._impl = backend.Workspace('')
        self._handle_pool = Workspace.HandlePool(self)

    def as_default(self):
        """Set as the default workspace.

        Call this method with the ``with`` keyword.

        Once ``with`` is exited, the previous default will be set.

        Returns
        -------
        dragon.Workspace
            This workspace.

        """
        return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_controller(self)

    def clear(self):
        """Release the tensors, operators and graphs."""
        self._impl.Clear()

    def create_graph(self, graph_def):
        """Create a graph."""
        cfg = config.config()
        if cfg.graph_verbosity == 2:
            msg = '\n' + str(graph_def)[:-1]
            logging.info('\ngraph {' + msg.replace('\n', '\n  ') + '\n}\n')
        return self._impl.CreateGraph(
            serialization.serialize_proto(graph_def),
            cfg.graph_verbosity == 1)

    def create_handle(self, key):
        """Create a handle."""
        return self._handle_pool.create(key)

    def create_tensor(self, name=None, scope=None):
        """Create a tensor."""
        if scope is not None:
            name = self.create_handle(scope)
        return self._impl.CreateTensor(name)

    def get_tensor(self, name):
        """Return the tensor."""
        return self._impl.GetTensor(name)

    def memory_allocated(self, device_type='cpu', device_index=0):
        """Return the size of device memory used by tensors.

        Parameters
        ----------
        device_type : str, optional
            The device type.
        device_index : int, optional
            The device index.

        Returns
        -------
        int
            The length of allocated bytes.

        """
        return self._impl.MemoryAllocated(device_type, device_index)

    def merge_from(self, other):
        """Merge resources from the other.

        Parameters
        ----------
        other : dragon.Workspace
            The workspace to merge.

        Returns
        -------
        dragon.Workspace
            This workspace.

        """
        self._impl.MergeFrom(other._impl)
        return self

    def release_handle(self, handle):
        """Release a handle."""
        self._handle_pool.release(handle)

    def run_backward(self, op_defs, targets, grad_targets=None, sources=None):
        """Compute the gradients of operators."""
        cfg = config.config()
        self._impl.RunBackward(
            op_defs,
            targets,
            grad_targets if grad_targets else [],
            sources if sources else [],
            cfg.graph_optimization > 2,
            cfg.graph_verbosity > 0,
        )

    def run_graph(self, name, execution_stage=None):
        """Run a graph."""
        stage_str = execution_stage if execution_stage else 'default'
        exec_stage = _PREDEFINED_GRAPH_EXECUTING_STAGES[stage_str]
        self._impl.RunGraph(name, exec_stage['include'], exec_stage['exclude'])

    def run_operator(self, op_def):
        """Run an operator."""
        cfg = config.config()
        if isinstance(op_def, dragon_pb2.OperatorDef):
            op_def = op_def.SerializePartialToString()
        self._impl.RunOperator(op_def, cfg.graph_verbosity > 0)

    def set_alias(self, target, alias):
        """Set an alias for the target."""
        self._impl.SetAlias(_stringify_object(target), alias)

    def unique_name(self, name, suffix='', namespace='', zero_based=True):
        """Return an unique name."""
        return self._impl.UniqueName(name, suffix, namespace, zero_based)


def get_workspace():
    """Return the default workspace.

    Returns
    -------
    dragon.Workspace
        The workspace.

    """
    return _GLOBAL_DEFAULT_WORKSPACE_STACK.get_default()


def reset_workspace():
    """Reset the default workspace."""
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
    'backward': {'include': '.*Gradient.*', 'exclude': 'GradientFill'},
    'backward_v2': {'include': '.*Gradient.*', 'exclude': ''},
}
