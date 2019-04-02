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

import math
import numpy

from dragon.ops import Run as _RunOp
from dragon.core import workspace as _workspace
from dragon.vm.tensorflow.framework import ops


class _DecayBase(object):
    def __init__(self):
        self.param_str = ''

    def set(self, tensor, value, dtype=None):
        _workspace.FeedTensor(tensor,
            value, dtype=dtype, force_cpu=True)

    def get(self, tensor):
        return _workspace.FetchTensor(tensor)


class _PiecewiseConstant(_DecayBase):
    def __init__(self):
        super(_PiecewiseConstant, self).__init__()

    def setup(self, *args, **kwargs):
        arguments = eval(self.param_str)
        self.boundaries = arguments['boundaries']
        self.values = arguments['values']

    def run(self, inputs, outputs):
        gs = self.get(inputs[0])
        for at in range(len(self.boundaries) - 1, -1, -1):
            if gs >= self.boundaries[at]:
                self.set(outputs[0], self.values[at + 1], dtype='float32')
                return


class _ExponentialDecay(_DecayBase):
    def __init__(self):
        super(_ExponentialDecay, self).__init__()

    def setup(self, *args, **kwargs):
        arguments = eval(self.param_str)
        self.learning_rate = arguments['learning_rate']
        self.decay_steps = arguments['decay_steps']
        self.decay_rate = arguments['decay_rate']
        self.staircase  = arguments['staircase']

    def run(self, inputs, outputs):
        gs = self.get(inputs[0])
        f = gs // self.decay_steps if self.staircase \
            else float(gs) / self.decay_steps
        new_lr = self.learning_rate * (self.decay_rate ** f)
        self.set(outputs[0], new_lr, dtype='float32')


class _NaturalExpDecay(_DecayBase):
    def __init__(self):
        super(_NaturalExpDecay, self).__init__()

    def setup(self, *args, **kwargs):
        arguments = eval(self.param_str)
        self.learning_rate = arguments['learning_rate']
        self.decay_steps = arguments['decay_steps']
        self.decay_rate = arguments['decay_rate']
        self.staircase  = arguments['staircase']

    def run(self, inputs, outputs):
        gs = self.get(inputs[0])
        f = gs // self.decay_steps if self.staircase \
            else float(gs) / self.decay_steps
        new_lr = self.learning_rate * math.exp(-self.decay_rate * f)
        self.set(outputs[0], new_lr, dtype='float32')


class _CosineDecay(_DecayBase):
    def __init__(self):
        super(_CosineDecay, self).__init__()

    def setup(self, *args, **kwargs):
        arguments = eval(self.param_str)
        self.learning_rate = arguments['learning_rate']
        self.decay_steps = arguments['decay_steps']
        self.alpha = arguments['alpha']

    def run(self, inputs, outputs):
        gs = self.get(inputs[0])
        global_step = min(gs, self.decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / self.decay_steps))
        decayed = (1. - self.alpha) * cosine_decay + self.alpha
        new_lr = self.learning_rate * decayed
        self.set(outputs[0], new_lr, dtype='float32')


class _CosineDecayRestarts(_DecayBase):
    def __init__(self):
        super(_CosineDecayRestarts, self).__init__()

    def setup(self, *args, **kwargs):
        arguments = eval(self.param_str)
        self.learning_rate = arguments['learning_rate']
        self.last_steps = 0
        self.decay_steps = arguments['first_decay_steps']
        self.t_mul, self.m_mul = arguments['t_mul'], arguments['m_mul']
        self.alpha = arguments['alpha']

    def run(self, inputs, outputs):
        gs = self.get(inputs[0])
        global_step = gs - self.last_steps
        cosine_decay = 0.5 * (1. + math.cos(
            math.pi * global_step / self.decay_steps))
        decayed = (1. - self.alpha) * cosine_decay + self.alpha
        new_lr = self.learning_rate * decayed
        # Restarts
        if global_step == self.decay_steps:
            self.last_steps = gs + 1
            self.decay_steps *= self.t_mul
            self.learning_rate *= self.m_mul
        self.set(outputs[0], new_lr, dtype='float32')


def piecewise_constant(
    x,
    boundaries,
    values,
    name=None,
):
    if len(values) != len(boundaries) + 1:
        raise ValueError('Excepted {} values, got {}.'.format(
            len(boundaries) + 1, len(values)))
    lr = _RunOp(
        inputs=[ops.convert_to_tensor(x)],
        module=__name__,
        op='_PiecewiseConstant',
        param_str=str({
            'boundaries': boundaries,
            'values': values,
        }),
        name=name,
    )
    lr.set_value(numpy.array(values[0], dtype='float32'))
    return lr


def exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None,
):
    lr = _RunOp(
        inputs=[ops.convert_to_tensor(global_step)],
        module=__name__,
        op='_ExponentialDecay',
        param_str=str({
            'learning_rate': learning_rate,
            'decay_steps': decay_steps,
            'decay_rate': decay_rate,
            'staircase': staircase,
        }),
        name=name,
    )
    lr.set_value(numpy.array(learning_rate, dtype='float32'))
    return lr


def natural_exp_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None,
):
    lr = _RunOp(
        inputs=[ops.convert_to_tensor(global_step)],
        module=__name__,
        op='_NaturalExpDecay',
        param_str=str({
            'learning_rate': learning_rate,
            'decay_steps': decay_steps,
            'decay_rate': decay_rate,
            'staircase': staircase,
        }),
        name=name,
    )
    lr.set_value(numpy.array(learning_rate, dtype='float32'))
    return lr


def cosine_decay(
    learning_rate,
    global_step,
    decay_steps,
    alpha=0.0,
    name=None,
):
    lr = _RunOp(
        inputs=[ops.convert_to_tensor(global_step)],
        module=__name__,
        op='_CosineDecay',
        param_str=str({
            'learning_rate': learning_rate,
            'decay_steps': decay_steps,
            'alpha': alpha,
        }),
        name=name,
    )
    lr.set_value(numpy.array(learning_rate, dtype='float32'))
    return lr


def cosine_decay_restarts(
    learning_rate,
    global_step,
    first_decay_steps,
    t_mul=2.0,
    m_mul=1.0,
    alpha=0.0,
    name=None,
):
    lr = _RunOp(
        inputs=[ops.convert_to_tensor(global_step)],
        module=__name__,
        op='_CosineDecayRestarts',
        param_str=str({
            'learning_rate': learning_rate,
            'first_decay_steps': first_decay_steps,
            't_mul': t_mul,
            'm_mul': m_mul,
            'alpha': alpha
        }),
        name=name,
    )
    lr.set_value(numpy.array(learning_rate, dtype='float32'))
    return lr


# Alias
piecewise_constant_decay = piecewise_constant