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

from dragon.core import tls as _tls
from dragon.core import scope as _scope
from dragon.core import workspace as _workspace
from dragon.core.tensor import Tensor as _Tensor

from dragon.vm.tensorflow.framework import constant_op


def convert_to_tensor(
    value,
    dtype=None,
    name=None,
    preferred_dtype=None,
):
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
    if isinstance(value, _Tensor): return value
    return constant_op.constant(value, dtype=dtype, name=name)


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
    return _scope.name_scope(name)


def get_default_graph():
    return _workspace.get_default_workspace()


def reset_default_graph():
    _workspace.reset_default_workspace()


def default_session(session):
    return _default_session_stack.get_controller(session)


def get_default_session():
    return _default_session_stack.get_default()


def device(device_name_or_function):
    if not isinstance(device_name_or_function, str):
        raise TypeError('The device function should be a str.')
    device_and_id = device_name_or_function.split('/')[1]
    device, id = device_and_id.split(':')
    if device not in ['cpu', 'gpu']:
        raise ValueError('The device should either be cpu or gpu.')
    try:
        id = int(id)
    except Exception as _:
        raise ValueError('The device id should be a integer.')
    return _scope.device_scope(device, device_id=id)


def _eval_using_default_session(tensors, feed_dict, session=None):
    if session is None:
        session = get_default_session()
        if session is None:
            raise ValueError("Cannot evaluate tensor using `eval()`: No default "
                             "session is registered. Use `with "
                             "sess.as_default()` or pass an explicit session to "
                             "`eval(session=sess)`")
    return session.run(tensors, feed_dict)


_default_session_stack = _tls.Stack()


# The Monkey Patching
# Require "import dragon.vm.tensorflow"
_Tensor.eval = lambda self, feed_dict=None, session=None : \
    _eval_using_default_session(self, feed_dict, session)