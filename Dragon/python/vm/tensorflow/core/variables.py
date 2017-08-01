# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = [
    'initialize_all_variables',
    'trainable_variables',
    'placeholder',
    'get_variable',
    'Variable'
]

from copy import deepcopy
from dragon.core.tensor import Tensor

VARIABLES = {}
TRAINABLE_VARIABLES = {}

def initialize_all_variables():
    outputs = []
    for tensor, initializer in VARIABLES.items():
        outputs.append(initializer)
    return outputs

def trainable_variables():
    return TRAINABLE_VARIABLES.values()

class placeholder(Tensor):
    def __init__(self, dtype, shape=None, name=None):
        super(placeholder, self).__init__(name, shape)
        self.Variable()  # register in dragon
        self.dtype = dtype


def get_variable(name, shape=None, initializer=None, trainable=True):
    from dragon.core.scope import TENSOR_SCOPE
    global VARIABLES, TENSOR_SCOPE

    name = TENSOR_SCOPE + name
    if not name in VARIABLES:
        # create a new variable
        if shape is None:
            raise ValueError('new Tensor({}) must specific a shape'.format(name))
        initial_value = initializer(shape)
        initial_value.expressions.values()[0].output[0] = name
        initial_value._name = name
        return Variable(initial_value, trainable)
    else:
        # return a copy if existing
        variable = deepcopy(VARIABLES[name])
        variable.expressions = {}
        return variable


class Variable(Tensor):
    def __init__(self, initial_value, trainable=True, name=None):
        super(Variable, self).__init__()
        from dragon.core.scope import TENSOR_SCOPE
        global VARIABLES, TRAINABLE_VARIABLES, TENSOR_SCOPE

        # initialize from a known Tensor
        if isinstance(initial_value, Tensor):
            self.clone(initial_value)
            if name is not None:
                name = TENSOR_SCOPE + name
                self._name = name
                initial_value.expressions.values()[0].output[0] = name
        else:
            from ..ops.constant_op import constant
            initial_value = constant(initial_value, name=name)
            self.clone(initial_value)

        # register in dragon
        self.Variable()

        # register in tf
        VARIABLES[self.name] = deepcopy(self)
        if trainable: TRAINABLE_VARIABLES[self.name] = deepcopy(self)

        # clear
        self.expressions = {}