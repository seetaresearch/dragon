# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Simple gradient maker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from dragon.core.autograph.op_def import OpDef
from dragon.core.framework import backend
from dragon.core.framework import proto_util
from dragon.core.proto import dragon_pb2


class GradientMaker(object):
    """The maker to generate grad defs to run backward."""

    @classmethod
    def gen_def(cls, op_def, grad_outputs):
        """Generate the grad def."""
        grad_defs, grad_inputs, defaults = backend.CreateGradientDef(
            op_def.SerializeToString(), grad_outputs)
        for i, grad_def in enumerate(grad_defs):
            new_def = dragon_pb2.OperatorDef()
            new_def.ParseFromString(grad_def)
            grad_defs[i] = new_def
        return grad_defs, grad_inputs, defaults

    @classmethod
    def check(cls, op_def, inputs_to_grads, targets):
        """Check if missing gradients. If missing, skip."""
        if op_def.type in backend.NO_GRADIENT_OPERATORS:
            return True, []
        gen_grads, maybe_skip = [], False
        for i, output in enumerate(op_def.output):
            if output not in inputs_to_grads:
                maybe_skip = True
                if output in targets:
                    gen_grads.append((output, i))
                    inputs_to_grads[output] = output + '_grad'
        return maybe_skip and len(gen_grads) == 0 and len(op_def.output) == 1, gen_grads

    @classmethod
    def make(cls, op_defs, targets, input_grads=None):
        """Make the grad defs."""
        inputs_to_grads = {} if input_grads is None else input_grads
        inputs_count, grads_count = defaultdict(int), defaultdict(int)

        # PLAY for the forward.
        for op_def in op_defs:
            if op_def.type in backend.NO_GRADIENT_OPERATORS:
                continue
            outputs = [output for output in op_def.output]
            for input in op_def.input:
                if input not in outputs:
                    # Avoid to count the duplicate input (i.e. the in-place output).
                    inputs_count[input] += 1

        # PLAY for the backward.
        backward_defs, split_grads = [], set()
        for op_def in op_defs[::-1]:
            # Generate def by registered gradient maker.
            is_skip, gen_grads = cls.check(op_def, inputs_to_grads, targets)
            grad_outputs = [inputs_to_grads.get(name, '') for name in op_def.output]
            grad_defs, grad_inputs, defaults = cls.gen_def(op_def, grad_outputs)

            # Add defs.
            if not is_skip:
                for input, grad_input in zip(op_def.input, grad_inputs):
                    inputs_to_grads[input] = grad_input
                # Add ``GradientGenerateOp``
                if len(gen_grads) > 0:
                    inputs, outputs, values = [], [], []
                    for name, i in gen_grads:
                        inputs.append(name)
                        outputs.append(name + '_grad')
                        values.append(defaults[i])
                    gen_op = proto_util.make_operator_def(
                        name=OpDef.get_name(),
                        op_type='GradientGenerate',
                        inputs=inputs,
                        outputs=outputs,
                        defaults=values,
                        device_option=op_def.device_option
                        if op_def.HasField('device_option') else None)
                    backward_defs.append(gen_op)
                # Add ``GradientOp``
                for grad_def in grad_defs:
                    grad_def.name = OpDef.get_name()
                    backward_defs.append(grad_def)

            # Split and gather gradient for multi-used inputs.
            for grad_def in grad_defs:
                for i, grad_name in enumerate(grad_def.output):
                    original_index = -1
                    for j, name in enumerate(grad_inputs):
                        if grad_name == name:
                            original_index = j
                    if original_index == -1 or grad_name in grad_def.input:
                        continue
                    original_name = op_def.input[original_index]
                    if inputs_count[original_name] <= 1:
                        continue
                    # Detect a split branch.
                    grad_name_v2 = grad_name + '_autosplit_%d' % grads_count[grad_name]
                    if not is_skip:
                        split_grads.add(grad_name_v2)
                    grads_count[grad_name] += 1
                    if grads_count[grad_name] == inputs_count[original_name]:
                        gather_inputs = []
                        for j in range(grads_count[grad_name]):
                            if '%s_autosplit_%d' % (grad_name, j) in split_grads:
                                gather_inputs.append('%s_autosplit_%d' % (grad_name, j))
                        backward_defs.append(proto_util.make_operator_def(
                            name=OpDef.get_name(),
                            op_type='GradientGather',
                            inputs=gather_inputs,
                            outputs=[grad_name],
                            device_option=grad_def.device_option
                            if grad_def.HasField('device_option') else None))
                    grad_def.output[i] = grad_name_v2

        return backward_defs
