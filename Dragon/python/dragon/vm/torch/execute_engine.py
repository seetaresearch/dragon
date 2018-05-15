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

"""Currently we provide two executing engine, ``ONCE`` and ``PERSISTENT``.

The basic idea of ``ONCE`` comes from ``caffe2``,
it spends much more time on Python frontend than C++ backend,
which should not be taken for running computation-intensive operators.

We extend a new ``PERSISTENT`` engine, that hashes the arguments
as many as possible, i.e., building with static operators while running
with dynamic arguments(e.g. inputs, outputs).

Note that not all operators can be re-written into the persistent mode,
we found it non-trivial to hash float arguments.
For those operators we still construct, run, and deconstruct them once.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon as dg
import dragon.core.utils as pb
from dragon.config import option

from .autograd.expression import Expression
from .autograd.grad_mode import is_grad_enabled
from .constants import CTX_TO_DEVICE_OPTION


def RunOperator(inputs, outputs, meta, auto_grad=True, **kwargs):
    if not isinstance(inputs, list): inputs = [inputs]
    if not isinstance(outputs, list): outputs = [outputs]
    if len(outputs) == 0:
        raise ValueError('The num of outputs should be at least 1.')


    inputs_name = [input.name for input in inputs]
    outputs_name = [output.name for output in outputs]

    engine_type = meta[0]; persistent_key = None

    if engine_type == 'ONCE':
        # OpType + CTX -> Op
        op_type, ctx = meta[1:]
        if ctx is None: raise ValueError('Excepted a context, got None.')
        op = pb.MakeOperatorDef(op_type, inputs_name, outputs_name, name='runtime',
                device_option=CTX_TO_DEVICE_OPTION[ctx], **kwargs)
    elif engine_type == 'PERSISTENT':
        # Key + Inputs + Outputs -> Op
        persistent_key, meta_op = meta[1:]
        op = pb.MutableOperatorDef(meta_op, inputs_name, outputs_name)
    else:
        raise ValueError('Unknown executing engine: {}.'.format(engine_type))

    # ~ Auto-Grad ~
    requires_grad = False
    for input in inputs:
        if input.requires_grad:
            requires_grad = True

    if len(inputs) > 0 and auto_grad:
        input_expressions = []
        if requires_grad and is_grad_enabled():
            ignored_grads = set()
            # + Trace outputs
            for input in inputs:
                input_expressions.append(input._expr)
                if input._ignored_grads:
                    ignored_grads = ignored_grads.union(input._ignored_grads)
            expression = Expression()
            expression.merge(input_expressions)
            op_name = expression.append(op)
            op.name = op_name
            for ix in range(len(outputs)):
                outputs[ix]._requires_grad = True
                outputs[ix]._expr = expression
                if len(ignored_grads) > 0:
                    outputs[ix]._ignored_grads = ignored_grads
        else:
            # + Reset status
            for ix in range(len(outputs)):
                outputs[ix]._requires_grad = False

    # Run
    if option['log_optimized_graph'] or option['log_meta_graph']:
        from dragon.config import logger
        logger.info('>>>>>>>>>>>>>>>>>> Forward Flow <<<<<<<<<<<<<<<<<<\n')
        logger.info(op)

    if engine_type == 'ONCE':
        dg.workspace.RunOperator(op)
    elif engine_type == 'PERSISTENT':
        dg.workspace.RunPersistentOp(persistent_key,
            op.name, inputs_name, outputs_name)

    # Returns
    if len(outputs) > 1: return outputs
    elif len(outputs) == 1: return outputs[0]
    else: return None