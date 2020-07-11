# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/caffe2/caffe2/blob/master/caffe2/python/core.py>
#
# ------------------------------------------------------------

"""Python-implemented gradient maker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from dragon import backend
from dragon.core.autograph.op_def import OpDef
from dragon.core.framework import proto_util
from dragon.core.proto import dragon_pb2


class GradientMaker(object):
    """Make def for the gradient based on rules."""

    @classmethod
    def gen_def(cls, op_def, g_outputs):
        """Generate the OperatorDef from forward op."""
        grad_defs, g_inputs, defaults = backend.CreateGradientDefs(
            op_def.SerializeToString(), g_outputs)
        for i, grad_def in enumerate(grad_defs):
            new_def = dragon_pb2.OperatorDef()
            new_def.ParseFromString(grad_def)
            grad_defs[i] = new_def
        return grad_defs, g_inputs, defaults

    @classmethod
    def check(cls, op_def, inputs_to_grads, blacklist, targets):
        """Check if missing gradients. If missing, skip."""
        if op_def.type in backend.NO_GRADIENT_OPERATORS:
            for input in op_def.input:
                blacklist.add(input)
            return True, None
        gen_grads = []
        for idx, output in enumerate(op_def.output):
            if output not in inputs_to_grads:
                if output in blacklist:
                    return True, gen_grads
                if output in targets:
                    # Consider to generate virtual gradient for targets.
                    gen_grads.append((output, idx))
                    inputs_to_grads[output] = output + '_grad'
                elif len(op_def.output) == 1:
                    # We can skip this op, obviously.
                    return True, gen_grads
        # Pass, even if missing some grads.
        return False, gen_grads

    @classmethod
    def make(cls, op_defs, targets, input_grads=None):
        """Make the backward op defs."""
        inputs_to_grads = {} if input_grads is None else input_grads
        inputs_count, grads_count = defaultdict(int), defaultdict(int)
        all_split_grads, blacklist = set(), set()

        # PLAY for the forward.
        for op_def in op_defs:
            if op_def.type in backend.NO_GRADIENT_OPERATORS:
                continue
            outputs = [output for output in op_def.output]
            for input in op_def.input:
                if input not in outputs:
                    # Avoid to count the duplicate input,
                    # (i.e. the in-place output).
                    inputs_count[input] += 1

        # PLAY for the backward.
        backward_defs = []
        for op_def in op_defs[::-1]:
            # Collect inputs and outputs.
            is_skip, gen_grads = cls.check(
                op_def=op_def,
                inputs_to_grads=inputs_to_grads,
                blacklist=blacklist,
                targets=targets,
            )
            # Missing grads are represented as ``None``.
            g_outputs = [inputs_to_grads.get(name, '') for name in op_def.output]
            grad_defs, g_inputs, defaults = cls.gen_def(op_def, g_outputs)

            # Append operators.
            if not is_skip:
                # GradientGenerateOp
                if len(gen_grads) > 0:
                    op_inputs, op_outputs, values = [], [], []
                    for item in gen_grads:
                        op_inputs.append(item[0])
                        op_outputs.append(item[0] + '_grad')
                        values.append(defaults[item[1]])
                    gen_op = proto_util.make_operator_def(
                        name=OpDef.get_name(),
                        op_type='GradientGenerate',
                        inputs=op_inputs,
                        outputs=op_outputs,
                        defaults=values,
                    )
                    if op_def.HasField('device_option'):
                        gen_op.device_option.CopyFrom(op_def.device_option)
                    backward_defs.append(gen_op)
                # GradientOp
                for grad_def in grad_defs:
                    grad_def.name = OpDef.get_name()
                    backward_defs.append(grad_def)

            # Split and gather grads for multi-used input.
            for grad_def in grad_defs:
                for g_output_idx, g_output in enumerate(grad_def.output):
                    original_idx = -1
                    for g_input_idx, g_input in enumerate(g_inputs):
                        if g_output == g_input:
                            original_idx = g_input_idx
                    # Ignore un-used && in-placed GI(?).
                    if original_idx == -1:
                        continue
                    if g_output in grad_def.input:
                        continue
                    # Found a split branch.
                    original_name = op_def.input[original_idx]
                    if inputs_count[original_name] > 1:
                        # Split.
                        split_name = g_output + '_autosplit_%d' % grads_count[g_output]
                        if not is_skip:
                            all_split_grads.add(split_name)
                        grads_count[g_output] += 1
                        # Gather.
                        if grads_count[g_output] == inputs_count[original_name]:
                            split_inputs = []
                            for idx in range(grads_count[g_output]):
                                if '%s_autosplit_%d' % (g_output, idx) in all_split_grads:
                                    split_inputs.append('%s_autosplit_%d' % (g_output, idx))
                            gather_def = proto_util.make_operator_def(
                                name=OpDef.get_name(),
                                op_type='GradientGather',
                                inputs=split_inputs,
                                outputs=[g_output],
                            )
                            if grad_def.HasField('device_option'):
                                gather_def.device_option.CopyFrom(grad_def.device_option)
                            backward_defs.append(gather_def)
                        grad_def.output[g_output_idx] = split_name

            # Done.
            if not is_skip:
                for name, grad in zip(op_def.input, g_inputs):
                    if grad != '':
                        inputs_to_grads[name] = grad

        return backward_defs
