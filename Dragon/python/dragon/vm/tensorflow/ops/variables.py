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

import copy

from dragon.core import scope as _scope
from dragon.core import workspace as _workspace
from dragon.core.tensor import Tensor as _Tensor
from dragon.vm.theano.compile import function as _Function

from dragon.vm.tensorflow.framework import ops, constant_op
from dragon.vm.tensorflow.util.deprecation import deprecated


class Variable(_Tensor):
    """Construct a Variable."""

    def __init__(
        self,
        initial_value=None,
        trainable=True,
        collections=None,
        validate_shape=True,
        name=None,
        dtype=None,
        regularizer=None,
        **kwargs
    ):
        super(Variable, self).__init__()

        if initial_value is None:
            raise ValueError('initial_value must be specified.')

        if collections is None:
            collections = [ops.GraphKeys.GLOBAL_VARIABLES]

        if not isinstance(collections, (list, tuple, set)):
            raise ValueError(
                'collections argument to Variable constructor must be a list, tuple, '
                    'or set. Got the type {}'.format(type(collections)))

        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

        if name is not None:
            # Get a known name from the name scope
            defined_name = _scope.get_default_name_scope() + name
        else:
            if 'name_from_variable_scope' in kwargs:
                # Has a name from the variable scope
                defined_name = kwargs['name_from_variable_scope']
            else:
                # Get a auto name from the name scope
                defined_name = _scope.get_default_name_scope() + 'Variable'

        # Set the name explicitly
        self.set_name(_workspace.GetDummyName(
            defined_name, suffix=':0', domain='Tensor'))

        # Initializer
        if isinstance(initial_value, _Tensor) and \
            len(initial_value.expressions) == 1:
                # From a initializing ops
                self.shape, self.dtype = \
                    initial_value.shape[:], \
                        initial_value.dtype
                init_expr = copy.deepcopy(initial_value.expressions)
                for k, v in init_expr.items():
                    init_expr[k].output[0] = self.name
                self.__init_expr__ = init_expr
        else:
            # From a const tensor
            if not isinstance(initial_value, _Tensor):
                initial_value = constant_op.constant(
                    initial_value, name=name, dtype=dtype)
            self.set_value(initial_value.get_value())
            self.shape, self.dtype = \
                initial_value.shape, \
                    initial_value.dtype

        # Regularizer
        self.__regularizer__ = regularizer

        # Registration
        self.Variable()

        if validate_shape:
            initial_value_shape = self.shape
            if initial_value_shape is None:
                raise ValueError('initial_value must have a shape specified.')

        ops.add_to_collections(collections, self)


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
        self.var_list = []
        for var in var_list:
            if hasattr(var, '__init_expr__'):
                var_copy = copy.deepcopy(var)
                var_copy.expressions = var.__init_expr__
                self.var_list.append(var_copy)

    def run(self):
        if not hasattr(self, '_init_func'):
            self._init_func = _Function(
                outputs=self.var_list) \
                    if len(self.var_list) > 0 else None
        if self._init_func: self._init_func()


def variables_initializer(var_list, name="init"):
    return VariablesInitializer(var_list)


def global_variables_initializer():
    return variables_initializer(global_variables())


@deprecated("2017-03-02", "Use `tf.global_variables_initializer` instead.")
def initialize_all_variables():
    """See ``tf.global_variables_initializer``."""
    return global_variables_initializer()