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

from dragon.vm.torch.autograd.anchor_pool import APool

# We simply use INC-UID to hash expressions,
# As Python supports BigInteger
_EXPRESSION_UID = 0


def _get_uid():
    global _EXPRESSION_UID
    _EXPRESSION_UID += 1
    return _EXPRESSION_UID - 1


class Expression(object):
    def __init__(self):
        self._ops = dict()

    def merge(self, expressions):
        for e in expressions:
            if e: self._ops.update(e._ops)

    def append(self, op):
        uid = _get_uid()
        op_name = APool.get(op.type)
        self._ops[uid] = op
        self._ops[uid].name = op_name
        return op_name

    def debug_str(self, name=''):
        external_inputs = set()
        ordered_ops = sorted(self._ops.items(), key=lambda d: d[0])
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