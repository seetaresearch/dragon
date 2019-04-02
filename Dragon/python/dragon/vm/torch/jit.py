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

"""A simple JIT expressions recorder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import tls as _tls
from dragon.vm.torch.c_api import _get_operator_pool


def _Incrementer():
    i = 0  # Python returns BigInteger
    while True: i += 1; yield i


class JITRecorder(object):
    UID_GENERATOR = _Incrementer()

    def __init__(self):
        self.ops = dict()

    def merge(self, recorders):
        for e in recorders:
            if e: self.ops.update(e.ops)

    def append(self, op):
        uid = next(self.UID_GENERATOR)
        op_name = _get_operator_pool().get(op.type)
        self.ops[uid] = op
        self.ops[uid].name = op_name
        return op_name

    def debug_str(self, name=''):
        external_inputs = set()
        ordered_ops = sorted(self.ops.items(), key=lambda d: d[0])
        outputs = set()
        buffer0 = '-------------------Expressions-------------------\n'
        buffer1 = ''
        buffer2 = 'Inputs: ['
        for k, v in ordered_ops:
            buffer1 = buffer1 + '>>>  ' + str(k) + '. ('
            for input in v.input:
                 if input not in outputs:
                    external_inputs.add(input)
                 buffer1 = buffer1 + input + ', '
            buffer1 = buffer1 + 'None, ' if len(v.input) == 0 else buffer1
            buffer1 = buffer1[0:-2] + ') -> ' + v.type + ' -> ('
            for output in v.output:
                outputs.add(output)
                buffer1 = buffer1 + output + ', '
            buffer1 = buffer1[0:-2] + ') \n'

        buffer1 = buffer1 + 'Target: ' + name + '\n'
        for ex_input in external_inputs:
            buffer2 = buffer2 + ex_input + ', '
        buffer2 = buffer2 + ']\n'
        return buffer0 + buffer2 + buffer1 + buffer0


def is_jit_enforced():
    """Whether jit tracer is enforced."""
    return _GLOBAL_ENFORCE_JIT_TRACER.enabled


class enforce_jit(object):
    """Context-manager that enforce the jit tracer."""

    def __init__(self):
        self.prev = is_jit_enforced()

    def __enter__(self):
        global _GLOBAL_ENFORCE_JIT_TRACER
        _GLOBAL_ENFORCE_JIT_TRACER.enabled = True

    def __exit__(self, *args):
        global _GLOBAL_ENFORCE_JIT_TRACER
        _GLOBAL_ENFORCE_JIT_TRACER.enabled = self.prev


_GLOBAL_ENFORCE_JIT_TRACER = _tls.Constant(enabled=False)