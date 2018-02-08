# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import warnings

from dragon.core.tensor import Tensor
import dragon.vm.theano as theano

from dragon.vm.tensorflow.protobuf import config_pb2
from dragon.vm.tensorflow.training.optimizer import Optimizer
from dragon.vm.tensorflow.ops.variables import VariablesInitializer
from dragon.vm.tensorflow.framework import ops


_TRANSACTIONS = {}


class Transaction(object):
    def __init__(self, functions):
        self.functions = functions

    def run(self, feed_values=None):
        for i, function in enumerate(self.functions):
            if i == 0 and feed_values is not None:
                function(*feed_values, return_outputs=False)
            else: function(return_outputs=False)

_default_session = None

class BaseSession(object):
    """
    Construct a BaseSession.
    """
    def __init__(self, target='', graph=None, config=None):
        if graph is None:
            self._graph = ops.get_default_graph()
        else:
            raise NotImplementedError('Session can only use the default graph yet.')

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

        self._session = None

    def list_devices(self):
        """Lists available devices in this session.

        ```python
        devices = sess.list_devices()
        for d in devices:
          print(d.name)
        ```

        Each element in the list has the following properties:
         - `name`: A string with the full name of the device. ex:
              `/job:worker/replica:0/task:3/device:CPU:0`
         - `device_type`: The type of the device (e.g. `CPU`, `GPU`, `TPU`.)
         - `memory_limit`: The maximum amount of memory available on the device.
              Note: depending on the device, it is possible the usable memory could
              be substantially less.
        Raises:
          tf.errors.OpError: If it encounters an error (e.g. session is in an
          invalid state, or network errors occur).

        Returns:
          A list of devices in the session.
        """
        # with errors.raise_exception_on_not_ok_status() as status:
        #     if self._created_with_new_api:
        #         raw_device_list = tf_session.TF_SessionListDevices(
        #             self._session, status)
        #     else:
        #         raw_device_list = tf_session.TF_DeprecatedSessionListDevices(
        #             self._session, status)
        #     device_list = []
        #     size = tf_session.TF_DeviceListCount(raw_device_list)
        #     for i in range(size):
        #         name = tf_session.TF_DeviceListName(raw_device_list, i, status)
        #         device_type = tf_session.TF_DeviceListType(raw_device_list, i, status)
        #         memory = tf_session.TF_DeviceListMemoryBytes(raw_device_list, i, status)
        #         device_list.append(_DeviceAttributes(name, device_type, memory))
        #     tf_session.TF_DeleteDeviceList(raw_device_list)
        #     return device_list

    def close(self):
        pass

    @property
    def graph(self):
        return self._graph

    @property
    def graph_def(self):
        return ''

    @property
    def sess_str(self):
        return ''

    def as_default(self):
        pass

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

        # unpack opts and tensors
        opts = []; tensors = []
        for target in fetches:
            if isinstance(target, Optimizer): opts.append(target)
            elif isinstance(target, VariablesInitializer): tensors.extend(target.var_list)
            elif isinstance(target, Tensor): tensors.append(target)

        # find minimum solving targets
        targets = set()
        for t in tensors: targets.add(t)
        for opt in opts:
            for t in opt.objs: targets.add(t)

        targets = list(targets)

        # if existing a transaction before?
        global _TRANSACTIONS
        t_key = tuple(fetches + feed_dict.keys()) \
                if feed_dict is not None else tuple(fetches)
        transaction = None if not t_key in _TRANSACTIONS else _TRANSACTIONS[t_key]

        # cond.1: run by feeding
        if feed_dict is not None:
            # checking
            for key, value in feed_dict.items():
                if not isinstance(key, Tensor):
                    raise TypeError('The key of feed_dict key should be a Tensor.')
                if key.shape is not None:
                    if len(key.shape) != len(value.shape):
                        raise RuntimeError('The Tensor({}) was limited to {} dimensions, \
                                            while feed a value with {} dimensions.'.
                                                format(key.name, len(key.shape), len(value.shape)))
                    for i in xrange(len(key.shape)):
                        if key.shape[i] is None: continue
                        if key.shape[i] != value.shape[i]:
                            raise RuntimeError('The shape of Tensor({}) was limited as ('.format(key.name) +
                                               ','.join([str(dim) for dim in key.shape]) + '), ' +
                                               'while feed a value with (' + ','.join([str(dim) for dim in value.shape]) + ').')
            # create a new transaction
            if transaction is None:
                functions = []
                functions.append(theano.function(inputs=feed_dict.keys(), outputs=targets))
                for opt in opts:
                    functions.append(theano.function(updater=opt.updater))
                _TRANSACTIONS[t_key] = transaction = Transaction(functions)
            transaction.run(feed_dict.values())

        # cond.2: run without feeding
        else:
            # create a new transaction
            if transaction is None:
                functions = []
                functions.append(theano.function(outputs=targets))
                for opt in opts:
                    functions.append(theano.function(updater=opt.updater))
                _TRANSACTIONS[t_key] = transaction = Transaction(functions)
            transaction.run(None)

        # fetch after running
        returns = []
        for target in fetches:
            if isinstance(target, Optimizer): returns.append(None)
            elif isinstance(target, VariablesInitializer): returns.append(None)
            else:
                np_target = target.get_value()
                # unpack the scalar if necessary
                if np_target.size == 1:
                    returns.append(np_target.flatten()[0])
                else:
                    returns.append(np_target)

        # unpack the returns if necessary
        if len(returns) == 1: return returns[0]
        else: return returns


class Session(BaseSession):
    """
    Construct a Session.
    """
    def __init__(self, target='', graph=None, config=None):
        super(Session, self).__init__(target, graph, config=config)

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        pass

    @staticmethod
    def reset(target, containers=None, config=None):
        pass


class InteractiveSession(BaseSession):
    """
    Construct a InteractiveSession.
    """
    def __init__(self, target='', graph=None, config=None):
        super(InteractiveSession, self).__init__(target, graph, config=config)

    def __enter__(self):
        pass

    def __exit__(self, exec_type, exec_value, exec_tb):
        pass

    @staticmethod
    def reset(target, containers=None, config=None):
        pass


def get_default_session():
    global _default_session
    if _default_session is None:
        _default_session = Session()
    return _default_session