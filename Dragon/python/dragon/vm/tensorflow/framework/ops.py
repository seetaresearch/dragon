# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import uuid
import threading
import dragon

from dragon.vm.tensorflow.framework import constant_op
from dragon.vm.tensorflow.util import tf_contextlib


def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    """Converts the given value to a Tensor.

    Parameters
    ----------
    value : number, sequence or numpy.ndarray
        The value to convert.
    dtype : DType, optional
        The data type. If ``None``, inferred from the type of ``value``.
    name : str, optional
        The Optional name.
    preferred_dtype : DType, optional
        The optional type when ``dtype`` is *None*.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if isinstance(value, dragon.Tensor): return value
    return constant_op.constant(value, dtype=dtype, name=name)


class Graph(object):
    """A wrapper to connect ``Function`` to ``Workspace``.

    Note that official TensorFlow trace the expressions explicitly
    in this class, while we have done in the virtual stack.

    Besides, organizing a ``Flow``, i.e., expressions with specified
    outputs should also be done here.

    """

    def __init__(self):
        self._collections = {}
        self._workspace = 'tf/graph/' + str(uuid.uuid4())

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

    def device(self, device_name_or_function):
        if not isinstance(device_name_or_function, str):
            raise TypeError('The device function should be a str.')
        device_and_id = device_name_or_function.split('/')[1]
        device, id = device_and_id.split(':')
        if device not in ['cpu', 'gpu']:
            raise ValueError('The device should either be cpu or gpu.')
        try:
            id = int(id)
        except Exception as e:
            raise ValueError('The device id should be a integer.')
        return dragon.device_scope(device, device_id=id)

    def as_default(self):
        return _default_graph_stack.get_controller(self)


class GraphKeys(object):
    GLOBAL_VARIABLES = "variables"
    # Key to collect local variables that are local to the machine and are not
    #  saved/restored.
    LOCAL_VARIABLES = "local_variables"
    # Key to collect model variables defined by layers.
    MODEL_VARIABLES = "model_variables"
    # Key to collect Variable objects that will be trained by the
    # optimizers.
    TRAINABLE_VARIABLES = "trainable_variables"
    # Key to collect summaries.
    SUMMARIES = "summaries"
    # Key to collect QueueRunners.
    QUEUE_RUNNERS = "queue_runners"
    # Key to collect table initializers.
    TABLE_INITIALIZERS = "table_initializer"
    # Key to collect asset filepaths. An asset represents an external resource
    # like a vocabulary file.
    ASSET_FILEPATHS = "asset_filepaths"
    # Key to collect Variable objects that keep moving averages.
    MOVING_AVERAGE_VARIABLES = "moving_average_variables"
    # Key to collect regularization losses at graph construction.
    REGULARIZATION_LOSSES = "regularization_losses"
    # Key to collect concatenated sharded variables.
    CONCATENATED_VARIABLES = "concatenated_variables"
    # Key to collect savers.
    SAVERS = "savers"
    # Key to collect weights
    WEIGHTS = "weights"
    # Key to collect biases
    BIASES = "biases"
    # Key to collect activations
    ACTIVATIONS = "activations"
    # Key to collect update_ops
    UPDATE_OPS = "update_ops"
    # Key to collect losses
    LOSSES = "losses"
    # Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
    SAVEABLE_OBJECTS = "saveable_objects"
    # Key to collect all shared resources used by the graph which need to be
    # initialized once per cluster.
    RESOURCES = "resources"
    # Key to collect all shared resources used in this graph which need to be
    # initialized once per session.
    LOCAL_RESOURCES = "local_resources"
    # Trainable resource-style variables.
    TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

    # Key to indicate various ops.
    INIT_OP = "init_op"
    LOCAL_INIT_OP = "local_init_op"
    READY_OP = "ready_op"
    READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
    SUMMARY_OP = "summary_op"
    GLOBAL_STEP = "global_step"

    # Used to count the number of evaluations performed during a single evaluation
    # run.
    EVAL_STEP = "eval_step"
    TRAIN_OP = "train_op"

    # Key for control flow context.
    COND_CONTEXT = "cond_context"
    WHILE_CONTEXT = "while_context"

    # Key for streaming model ports.
    # NOTE(yuanbyu): internal and experimental.
    _STREAMING_MODEL_PORTS = "streaming_model_ports"


def get_collection_ref(key):
    return get_default_graph().get_collection_ref(key)


def get_collection(key, scope=None):
    return get_default_graph().get_collection(key, scope)


def add_to_collection(name, value):
    get_default_graph().add_to_collection(name, value)


def add_to_collections(names, value):
    get_default_graph().add_to_collections(names, value)


def name_scope(name, default_name=None, values=None):
    name = default_name if name is None else name
    name = '' if name is None else name
    return dragon.name_scope(name)


##############################################
#                                            #
#              Default Stack                 #
#                                            #
##############################################


class _DefaultStack(threading.local):
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super(_DefaultStack, self).__init__()
        self._enforce_nesting = True
        self.stack = []

    def get_default(self):
        return self.stack[-1] if len(self.stack) >= 1 else None

    def reset(self):
        self.stack = []

    def is_cleared(self):
        return not self.stack

    @property
    def enforce_nesting(self):
        return self._enforce_nesting

    @enforce_nesting.setter
    def enforce_nesting(self, value):
        self._enforce_nesting = value

    @tf_contextlib.contextmanager
    def get_controller(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            # stack may be empty if reset() was called
            if self.stack:
                if self._enforce_nesting:
                    if self.stack[-1] is not default:
                        raise AssertionError(
                            "Nesting violated for default stack of %s objects" %
                            type(default))
                    self.stack.pop()
                else:
                    self.stack.remove(default)


class _DefaultGraphStack(_DefaultStack):
    """A thread-local stack of objects for providing an implicit default graph."""

    def __init__(self):
        super(_DefaultGraphStack, self).__init__()
        self._global_default_graph = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        ret = super(_DefaultGraphStack, self).get_default()
        if ret is None:
            ret = self._GetGlobalDefaultGraph()
        return ret

    def _GetGlobalDefaultGraph(self):
        if self._global_default_graph is None:
            # TODO(mrry): Perhaps log that the default graph is being used, or set
            #   provide some other feedback to prevent confusion when a mixture of
            #   the global default graph and an explicit graph are combined in the
            #   same process.
            self._global_default_graph = Graph()
            # Rewritten the random workspace name
            self._global_default_graph._workspace = 'default'
        return self._global_default_graph

    def reset(self):
        super(_DefaultGraphStack, self).reset()
        # We should call dragon api to reset the workspace
        dragon.workspace.ResetWorkspace(self._global_default_graph._workspace)
        self._global_default_graph = None

    @tf_contextlib.contextmanager
    def get_controller(self, default):
        with super(_DefaultGraphStack, self).get_controller(default) as g:
            with dragon.workspace_scope(g._workspace):
                yield g


_default_graph_stack = _DefaultGraphStack()
_default_session_stack = _DefaultStack()


def get_default_graph():
    return _default_graph_stack.get_default()


def reset_default_graph():
    if not _default_graph_stack.is_cleared():
        raise AssertionError("Do not use tf.reset_default_graph() to clear "
                             "nested graphs. If you need a cleared graph, "
                             "exit the nesting and create a new graph.")
    _default_graph_stack.reset()


def default_session(session):
    return _default_session_stack.get_controller(session)


def get_default_session():
    return _default_session_stack.get_default()


def device(device_name_or_function):
    return get_default_graph().device(device_name_or_function)


def _eval_using_default_session(tensors, feed_dict, session=None):
    if session is None:
        session = get_default_session()
        if session is None:
            raise ValueError("Cannot evaluate tensor using `eval()`: No default "
                             "session is registered. Use `with "
                             "sess.as_default()` or pass an explicit session to "
                             "`eval(session=sess)`")
    return session.run(tensors, feed_dict)


# Require "import dragon.vm.tensorflow"
dragon.Tensor.eval = lambda self, feed_dict=None, session=None : \
    _eval_using_default_session(self, feed_dict, session)