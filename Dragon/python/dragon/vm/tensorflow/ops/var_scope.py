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

import threading

from dragon.core import tls as _tls
from dragon.core import scope as _scope

from dragon.vm.tensorflow.framework import dtypes, ops
from dragon.vm.tensorflow.ops.variables import Variable
from dragon.vm.tensorflow.ops import init_ops


class VariableScope(object):
    """Construct a Variable."""

    def __init__(self, reuse, name='', name_scope='', **kwargs):
        # Whether to reuse the existing variables
        self._reuse = reuse
        # Store the variable name scope till the current level
        self._name = name
        # Store the tensor name scope till the current level
        self._name_scope = name_scope if name_scope else ''
        # A dictionary of the stored TensorFlow variables.
        self._vars = {}
        # Store the previous variable scope object
        self._old_scope = None
        # Store the name scope context manager
        self._name_scope_ctx = kwargs.get('name_scope_ctx', None)

    @property
    def reuse(self):
        """Whether this variable scope can reuse the variables.

        Returns
        -------
        boolean
            ``True`` if variables can be reused.

        """
        return self._reuse

    @property
    def name(self):
        """Return the tensor name scope till the current level.

        Returns
        -------
        str
            The tensor name scope.

        """
        return self._name

    @property
    def original_name_scope(self):
        """Return the variable name scope till the current level.

        Returns
        -------
        str
            The variable name scope.

        """
        return self._name_scope

    @property
    def vars(self):
        """Return the variable dict of this scope.

        Returns
        -------
        dict of Tensor
            The variable dict.

        """
        return self._vars

    def get_variable(
        self,
        name,
        shape=None,
        dtype=None,
        initializer=None,
        regularizer=None,
        trainable=True,
        collections=None,
        validate_shape=True,
    ):
        excepted_name = self.name + name
        if not excepted_name in self._vars:
            # Create a new variable
            if shape is None:
                raise ValueError(
                    'Must specific a shape to create a Variable.')
            if initializer is None:
                initializer = self._get_default_initializer(
                    name, shape=shape, dtype=dtype)
            variable = Variable(
                initial_value=initializer(shape, dtype=dtype),
                regularizer=regularizer,
                trainable=trainable,
                collections=collections,
                validate_shape=validate_shape,
                name_from_variable_scope=excepted_name,
                dtype=dtype,
            )
            self._vars[excepted_name] = variable
            return variable
        else:
            # Return a existing variable
            if self._reuse:
                return self._vars[excepted_name]
            raise ValueError('Variable {} already exists, disallowed. '
                'Did you mean to set reuse=True in VarScope?'.format(excepted_name))

    def __enter__(self):
        # Variable scope will also affect the global name scope
        self._name_scope_ctx.__enter__()
        get_variable_scope_store().open(self)
        return self

    def __exit__(self, type, value, traceback):
        get_variable_scope_store().close()
        self._name_scope_ctx.__exit__(type, value, traceback)

    def _get_default_initializer(
        self,
        name,
        shape=None,
        dtype=dtypes.float32,
    ):
        # Defaults: float32
        if dtype is None:
            dtype = dtypes.float32

        # Xavier for float16, float32, float64
        if dtype.is_floating:
            initializer = init_ops.glorot_uniform_initializer()

        # Zeros for integers
        elif dtype.is_integer or \
                dtype.is_unsigned or \
                    dtype.is_bool:
            initializer = init_ops.zeros_initializer()(
                shape=shape, dtype=dtype.base_dtype)

        # Fail to match the DType
        else:
            raise ValueError(
                'An initializer for Variable({}) of %s is required.'
                    .format(name, dtype.base_dtype))

        return initializer


def variable_scope(name_or_scope, reuse=None, **kwargs):
    name_or_scope = name_or_scope if name_or_scope else ''
    prefix = name_or_scope + '/' if name_or_scope != '' else ''
    vs_store = get_variable_scope_store()
    vs_name = vs_store.current_scope.name + prefix
    original_name_scope = _scope.get_default_name_scope() + prefix
    vs = VariableScope(reuse, name=vs_name, name_scope=original_name_scope)
    # Store the ctx manager instead of returning
    # As we should return a VariableScope
    vs._name_scope_ctx = _scope.name_scope(name_or_scope)
    return vs


def get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    **kwargs
):
    return get_variable_scope().get_variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
    )


class _VariableScopeStore(threading.local):
    """A thread local store for the current variable scope."""

    def __init__(self):
        super(_VariableScopeStore, self).__init__()
        self.name_scope = None
        self.previous_scope = None
        self.current_scope = VariableScope(False)

    def open(self, var_scope):
        self.previous_scope = self.current_scope
        self.current_scope = var_scope

    def close(self):
        self.current_scope = self.previous_scope


def get_variable_scope_store():
    scope_store = ops.get_collection(_GLOBAL_VARIABLE_SCOPE_STORE_KEY)
    if not scope_store:
        scope_store = _VariableScopeStore()
        ops.add_to_collection(_GLOBAL_VARIABLE_SCOPE_STORE_KEY, scope_store)
    else:
        scope_store = scope_store[0]
    return scope_store


def get_variable_scope():
    """Returns the current variable scope."""
    return get_variable_scope_store().current_scope


_GLOBAL_VARIABLE_SCOPE_STORE_KEY = ("__varscope",)
_GLOBAL_VARIABLE_SCOPE_STACK = _tls.Stack()