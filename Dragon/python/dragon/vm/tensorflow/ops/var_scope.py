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

from dragon.vm.tensorflow.framework import dtypes
from dragon.vm.tensorflow.ops.variables import Variable
from dragon.vm.tensorflow.ops import init_ops

_VARSCOPE = None
_VARSTORE = {}

class VariableScope(object):
    """
    Construct a Variable.
    """
    def __init__(self, reuse, name='', name_scope='', **kwargs):
        self._name = name
        self._reuse = reuse
        self._name_scope = name_scope
        if self._name_scope is None:
            self._name_scope = ''
        self._old_varscope = None

    @property
    def reuse(self):
        return self._reuse

    @property
    def name(self):
        return self._name

    @property
    def original_name_scope(self):
        return self._name_scope

    def get_variable(self, name, shape=None, dtype=None, initializer=None,
                     trainable=True, collections=None, validate_shape=True, **kwargs):
        global _VARSTORE

        # get full name
        from dragon.core.scope import get_tensor_scope
        full_name = get_tensor_scope() + name

        # create a new variable
        if not full_name in _VARSTORE:
            if shape is None:
                raise ValueError('Must specific a shape for the Variable({}).'.format(full_name))
            if initializer is None:
                initializer = self._get_default_initializer(name, shape=shape, dtype=dtype)
            initial_value = initializer(shape, dtype=dtype)
            new_var = Variable(initial_value, trainable=trainable, collections=collections,
                            validate_shape=validate_shape, name=name, dtype=dtype)
            _VARSTORE[full_name] = new_var
            return new_var
        else:
            # existing ?
            if self._reuse:
                return _VARSTORE[full_name]
            raise ValueError('The Variable({}) already exists.'.format(full_name))

    def __enter__(self):
        global _VARSCOPE
        self._old_varscope = _VARSCOPE
        _VARSCOPE = self

        from dragon.core.scope import get_tensor_scope, set_tensor_scope
        prefix = self._name_scope + '/' if self._name_scope != '' else ''
        set_tensor_scope(get_tensor_scope() + prefix)
        return self

    def __exit__(self, type, value, traceback):
        global _VARSCOPE
        _VARSCOPE = self._old_varscope

        from dragon.core.scope import get_tensor_scope, set_tensor_scope
        prefix = self._name_scope + '/' if self._name_scope != '' else ''
        assert get_tensor_scope().endswith(prefix)
        if self._name_scope != '':
            set_tensor_scope(get_tensor_scope()[:-len(prefix)])

    def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
        if dtype is None: dtype = dtypes.float32
        if dtype.is_floating:
            initializer = init_ops.glorot_uniform_initializer()
        elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
            initializer = init_ops.zeros_initializer()(shape=shape, dtype=dtype.base_dtype)
        else:
            raise ValueError('An initializer for Variable({}) of %s is required.'.
                             format(name, dtype.base_dtype))
        return initializer


def get_variable_scope():
    global _VARSCOPE
    if _VARSCOPE is None:
        _VARSCOPE = VariableScope(False)
    return _VARSCOPE


def variable_scope(name_scope, reuse=None, **kwargs):
    return VariableScope(reuse, name_scope=name_scope)


def get_variable(name, shape=None, dtype=None, initializer=None,
                 trainable=True, collections=None, validate_shape=True, **kwargs):
    return get_variable_scope().get_variable(name, shape=shape, dtype=dtype,
                                             initializer=initializer,trainable=trainable,
                                             collections=collections, validate_shape=validate_shape)