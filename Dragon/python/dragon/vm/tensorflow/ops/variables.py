# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import copy

from dragon.core.tensor import Tensor
import dragon.vm.theano as theano

from dragon.vm.tensorflow.framework import ops
from dragon.vm.tensorflow.framework import dtypes
from dragon.vm.tensorflow.util.deprecation import deprecated


class Variable(Tensor):
    """
    Construct a Variable.
    """
    def __init__(self, initial_value=None, trainable=True,
                 collections=None, validate_shape=True,
                 name=None, dtype=None, **kwargs):
        super(Variable, self).__init__()

        if initial_value is None:
            raise ValueError('initial_value must be specified.')

        if collections is None:
            collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        if not isinstance(collections, (list, tuple, set)):
            raise ValueError('collections argument to Variable constructor must be a list, tuple, '
                             'or set. Got the type {}'.format(type(collections)))
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

        # initialization
        if isinstance(initial_value, Tensor):
            self.clone(initial_value)
            if name is not None:
                self.name = name
                self.expressions.values()[0].output[0] = self.name
        else:
            # from ..ops.constant_op import constant
            # initial_value = constant(initial_value, name=name)
            # self.clone(initial_value)
            pass

        # check data type
        if dtype is not None:
            if not isinstance(dtype, dtypes.DType):
                raise TypeError('The dtype should be a valid tf data type.')
            self.dtype = dtype.name

        # registration
        self.Variable()

        if validate_shape:
            initial_value_shape = self.shape
            if initial_value_shape is None:
                raise ValueError('initial_value must have a shape specified.')

        ops.add_to_collections(collections, copy.deepcopy(self))
        self.expressions = {}


def global_variables():
    return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)


def local_variables():
    return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)


def model_variables():
    return ops.get_collection(ops.GraphKeys.MODEL_VARIABLES)


def trainable_variables():
    return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)


class VariablesInitializer(object):
    def __init__(self, var_list):
        self.var_list = var_list

    def run(self):
        if not hasattr(self, '_init_func'):
            self._init_func = theano.function(outputs=self.var_list)
        self._init_func()


def variables_initializer(var_list, name="init"):
    return VariablesInitializer(var_list)


def global_variables_initializer():
    return variables_initializer(global_variables())


@deprecated("2017-03-02", "Use `tf.global_variables_initializer` instead.")
def initialize_all_variables():
    """
    See ``tf.global_variables_initializer``.
    """
    return global_variables_initializer()
