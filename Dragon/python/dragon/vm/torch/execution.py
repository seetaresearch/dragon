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

"""The basic idea of directly run operators comes from ``caffe2``,
it spends much more time on Python frontend than C++ backend,
which should not be taken for running computation-intensive operators.

We extend a new ``PERSISTENT`` engine, that hashes the arguments
as many as possible, i.e., creates a operator once while running
with arbitrary inputs and outputs many times.

Note that it is still a challenge to persist the operators which
take the argument with uncertain numerical bounds. In this case,
our engine will still create lots of duplicates.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import dragon as dg
import dragon.import_c_api as C
from dragon.config import option

from .c_api import device as _Device
from .jit import JITRecorder, is_jit_enforced
from .autograd.grad_mode import is_grad_enabled
from .tensor import _RuntimeTensor
from .pool import TensorPool


def RunOperator(
    inputs, outputs, meta,
        auto_grad=True,
            callback_on_run=None):
    if not isinstance(inputs, list): inputs = [inputs]
    if not isinstance(outputs, list): outputs = [outputs]
    if len(outputs) == 0:
        raise ValueError('The num of outputs should be at least 1.')

    # I/O Check
    requires_grad = False
    inputs_name, outputs_name = [], []
    for input in inputs:
        inputs_name.append(input.name)
        if input.requires_grad: requires_grad = True

    requires_grad = \
        (requires_grad and
            is_grad_enabled()) or \
                is_jit_enforced()

    for ix, output in enumerate(outputs):
        if isinstance(output, six.string_types):
            # Lite mode, the name of output is given
            outputs_name.append(output)
        else:
            # Legacy mode, a torch tensor is excepted
            if isinstance(output, _Device):
                name = TensorPool.get('${JOIN}' if requires_grad else '${DETACH}')
                outputs[ix] = _RuntimeTensor(name, device=output)
            outputs_name.append(outputs[ix].name)

    # Key + Inputs + Outputs => Op
    op_name = 'runtime'
    persistent_key, meta_op = meta
    op = C.OperatorDef(); op.CopyFrom(meta_op)
    op.input, op.output = inputs_name, outputs_name

    # Auto-Grad
    if len(inputs) > 0 and auto_grad:
        input_recorders = []
        if requires_grad:
            ignored_grads = set()
            # Trace outputs
            for input in inputs:
                input_recorders.append(input.__jit_recorder__)
                if input._ignored_grads:
                    ignored_grads = ignored_grads.union(
                        input._ignored_grads)
            recorder = JITRecorder()
            recorder.merge(input_recorders)
            op_name = recorder.append(op)
            op.name = op_name
            for ix in range(len(outputs)):
                outputs[ix].requires_grad = True
                outputs[ix].__jit_recorder__ = recorder
                if len(ignored_grads) > 0:
                    outputs[ix]._ignored_grads = ignored_grads
        else:
            # Reset status
            for ix in range(len(outputs)):
                outputs[ix].requires_grad = False

    # Callback on Run
    if callback_on_run: callback_on_run(op_name)

    # Run
    dg.workspace.RunOperator(op,
        verbose=option['log_optimized_graph'] or
                option['log_meta_graph'])

    # Returns
    if len(outputs) > 1: return outputs
    elif len(outputs) == 1: return outputs[0]
    else: return None