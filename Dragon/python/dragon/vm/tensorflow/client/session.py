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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from collections import defaultdict

from dragon.core import workspace as _workspace
from dragon.core.tensor import Tensor as _Tensor
from dragon.vm.theano.compile import function as _Function
from dragon.vm.tensorflow.protobuf import config_pb2
from dragon.vm.tensorflow.training.optimizer import Optimizer
from dragon.vm.tensorflow.ops.variables import VariablesInitializer
from dragon.vm.tensorflow.framework import ops


class _DataFlow(object):
    """DataFlow takes a group of expressions and
    the specified output tensors.

    We store the flows that requiring the same output names,
    i.e., those flows can be reused and should not be created again.

    """
    def __init__(self, functions):
        self.functions = functions

    def run(self, feed_dict=None):
        for i, func in enumerate(self.functions):
            if i == 0 and feed_dict is not None:
                for tensor, value in feed_dict.items():
                    _workspace.FeedTensor(tensor, value)
            func(return_outputs=False)

    @classmethod
    def try_get(cls, graph_id, flow_key):
        if flow_key in _GLOBAL_DATA_FLOWS[graph_id]:
            return _GLOBAL_DATA_FLOWS[graph_id][flow_key]

    @classmethod
    def try_add(cls, graph_id, flow_key, flow):
        global _GLOBAL_DATA_FLOWS
        _GLOBAL_DATA_FLOWS[graph_id][flow_key] = flow


class BaseSession(object):
    """Construct a BaseSession."""

    def __init__(self, target='', graph=None, config=None):
        if graph is None:
            self._graph = ops.get_default_graph()
        else:
            self._graph = graph

        self._opened = False
        self._closed = False

        if config is not None:
            if not isinstance(config, config_pb2.ConfigProto):
                raise TypeError('Config should be a tf.ConfigProto, but got {}'.format(type(config)))
            self._config = config
            self._add_shapes = config.graph_options.infer_shapes
        else:
            self._config = None
            self._add_shapes = False

    def list_devices(self):
        pass

    def close(self):
        pass

    @property
    def graph(self):
        return self._graph

    def as_default(self):
        return ops.default_session(self)

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        try:
            if options is not None:
                warnings.warn(Warning('Ignore Arguments: <options>.'))
            if run_metadata is not None:
                warnings.warn(Warning('Ignore Arguments: <run_metadata>.'))
        except Warning:
            pass

        if not isinstance(fetches, list): fetches = [fetches]
        if len(fetches) < 1: return None

        return self._run(fetches, feed_dict)

    def _run(self, fetches, feed_dict):
        if self._closed:
            raise RuntimeError('Attempted to use a closed Session.')

        # Unpack opts and tensors
        tensors, optimizers = [], []

        for e in fetches:
            if isinstance(e, Optimizer): optimizers.append(e)
            elif isinstance(e, VariablesInitializer): tensors.extend(e.var_list)
            elif isinstance(e, _Tensor): tensors.append(e)

        # Find minimum solving targets
        targets = set()
        for e in tensors: targets.add(e)
        for optimizer in optimizers:
            for t in optimizer._targets: targets.add(t)

        targets = list(targets)
        flow_key = tuple(e.name for e in targets)

        # Exist this data flow before?
        flow = _DataFlow.try_get(id(self._graph), flow_key)

        # Run by feeding
        if feed_dict is not None:
            # Check the feed dict
            for key, value in feed_dict.items():
                if not isinstance(key, _Tensor):
                    raise TypeError('The key of ``feed_dict`` should be a Tensor.')
                if key.shape is not None:
                    # Align the number of dimensions
                    if len(key.shape) != len(value.shape):
                        raise RuntimeError(
                            'The Tensor({}) was limited to {} dimensions, '\
                            'while feed a value with {} dimensions.'
                            .format(key.name, len(key.shape), len(value.shape)))
                    # Verify for the each dimension
                    for i in range(len(key.shape)):
                        if key.shape[i] is None: continue
                        if key.shape[i] != value.shape[i]:
                            raise RuntimeError(
                                'The shape of Tensor({}) was limited as ('.format(key.name) +
                                ','.join([str(dim) for dim in key.shape]) + '), ' +
                                'while feed a value with (' +
                                ','.join([str(dim) for dim in value.shape]) + ').')

        # Create a new data flow if necessary
        if flow is None:
            functions = [_Function(outputs=targets)]
            for optimizer in optimizers:
                functions.append(_Function(
                    updater=optimizer.updater))
            flow = _DataFlow(functions)
            _DataFlow.try_add(id(self._graph), flow_key, flow)

        # Run this data flow
        flow.run(feed_dict)

        # Fetch after running
        returns = []

        for e in fetches:
            if isinstance(e, Optimizer):
                e._inc_global_step()
                returns.append(None)
            elif isinstance(e, VariablesInitializer):
                returns.append(None)
            else:
                np_target = e.get_value()
                # Unpack the scalar if necessary
                if np_target.size == 1: returns.append(np_target.flatten()[0])
                else: returns.append(np_target)

        # Unpack the returns if necessary
        if len(returns) == 1: return returns[0]
        else: return returns


class Session(BaseSession):
    """Construct a Session."""

    def __init__(self, target='', graph=None, config=None):
        super(Session, self).__init__(target, graph, config=config)
        self._default_graph_context_manager = None
        self._default_session_context_manager = None

    def __enter__(self):
        if self._default_graph_context_manager is None:
            self._default_graph_context_manager = self.graph.as_default()
        else:
            raise RuntimeError('Session context managers are not re-entrant. '
                               'Use `Session.as_default()` if you want to enter '
                               'a session multiple times.')
        if self._default_session_context_manager is None:
            self._default_session_context_manager = self.as_default()
        self._default_graph_context_manager.__enter__()
        return self._default_session_context_manager.__enter__()

    def __exit__(self, exec_type, exec_value, exec_tb):
        try:
            self._default_session_context_manager.__exit__(exec_type, exec_value, exec_tb)
        except RuntimeError as error:
            if error == exec_value:
                pass
            else:
                raise
        self._default_graph_context_manager.__exit__(exec_type, exec_value, exec_tb)

    @staticmethod
    def reset(target, containers=None, config=None):
        pass


class InteractiveSession(BaseSession):
    """Construct a InteractiveSession."""

    def __init__(self, target='', graph=None, config=None):
        super(InteractiveSession, self).__init__(target, graph, config=config)

    def __enter__(self):
        pass

    def __exit__(self, exec_type, exec_value, exec_tb):
        pass

    @staticmethod
    def reset(target, containers=None, config=None):
        pass


# Store the flows for different graphs
# ThreadLocal is not necessary
_GLOBAL_DATA_FLOWS = defaultdict(dict)