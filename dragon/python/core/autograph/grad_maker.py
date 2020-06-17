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

"""Gradient maker implemented in python.

The basic idea of ``GradientMaker`` comes from ``caffe2``,
Jia provided a simple way to bridge the Generator(Python) with OpScheme(C++).

For the efficient C++ implementation, see,

    <https://github.com/seetaresearch/Dragon/blob/master/Dragon/src/core/graph_gradient.cc>

"""

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
    def gen_def(cls, forward_op, g_outputs):
        """Generate the OperatorDef from forward op."""
        g_ops, g_inputs, defaults = backend.CreateGradientDefs(
            forward_op.SerializeToString(), g_outputs)
        for idx, g_op in enumerate(g_ops):
            new_def = dragon_pb2.OperatorDef()
            new_def.ParseFromString(g_op)
            g_ops[idx] = new_def
        return g_ops, g_inputs, defaults

    @classmethod
    def check(cls, forward_op, inputs_to_grads, blacklist, targets):
        """Check if missing gradients. If missing, skip."""
        if forward_op.type in backend.NO_GRADIENT_OPERATORS:
            for input in forward_op.input:
                blacklist.add(input)
            return True, None
        gen_grads = []
        for idx, output in enumerate(forward_op.output):
            if output not in inputs_to_grads:
                if output in blacklist:
                    return True, gen_grads
                if output in targets:
                    # Consider to generate virtual gradient for targets.
                    gen_grads.append((output, idx))
                    inputs_to_grads[output] = output + '_grad'
                elif len(forward_op.output) == 1:
                    # We can skip this op, obviously.
                    return True, gen_grads
        # Pass, even if missing some grads.
        return False, gen_grads

    @classmethod
    def make(cls, forward_ops, targets, input_grads=None):
        """The making procedure."""
        inputs_to_grads = {} if input_grads is None else input_grads
        inputs_count, grads_count = defaultdict(int), defaultdict(int)
        all_split_grads, blacklist = set(), set()

        backward_ops = []

        # A DAG may not have any in-place operators.
        is_dag = True

        # PLAY for the forward.
        for forward_op in forward_ops:
            if forward_op.type in backend.NO_GRADIENT_OPERATORS:
                continue
            outputs = [o for o in forward_op.output]
            for input in forward_op.input:
                if input not in outputs:
                    # Avoid to count the duplicate input,
                    # (i.e. the in-place output).
                    inputs_count[input] += 1
                else:
                    is_dag = False

        # PLAY for the backward.
        for forward_op in forward_ops[::-1]:
            # Collect inputs and outputs.
            is_skip, gen_grads = cls.check(
                forward_op=forward_op,
                inputs_to_grads=inputs_to_grads,
                blacklist=blacklist,
                targets=targets,
            )
            # Missing grads are represented as ``None``.
            g_outputs = [inputs_to_grads.get(name, '')
                         for name in forward_op.output]
            g_ops, g_inputs, defaults = cls.gen_def(forward_op, g_outputs)

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
                    if forward_op.HasField('device_option'):
                        gen_op.device_option.CopyFrom(forward_op.device_option)
                    backward_ops.append(gen_op)
                # GradientOp
                for g_op in g_ops:
                    g_op.name = OpDef.get_name()
                    backward_ops.append(g_op)

            # Split and gather grads for multi-used input.
            for g_op in g_ops:
                for g_output_idx, g_output in enumerate(g_op.output):
                    original_idx = -1
                    for g_input_idx, g_input in enumerate(g_inputs):
                        if g_output == g_input:
                            original_idx = g_input_idx
                    # Ignore un-used && in-placed GI(?).
                    if original_idx == -1:
                        continue
                    if g_output in g_op.input:
                        continue
                    # Found a split branch.
                    original_name = forward_op.input[original_idx]
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
                            gather_op = proto_util.make_operator_def(
                                name=OpDef.get_name(),
                                op_type='GradientGather',
                                inputs=split_inputs,
                                outputs=[g_output],
                            )
                            if g_op.HasField('device_option'):
                                gather_op.device_option.CopyFrom(g_op.device_option)
                            backward_ops.append(gather_op)
                        g_op.output[g_output_idx] = split_name

            # Done.
            if not is_skip:
                for name, grad in zip(forward_op.input, g_inputs):
                    if grad != '':
                        inputs_to_grads[name] = grad

        return forward_ops, backward_ops, is_dag
