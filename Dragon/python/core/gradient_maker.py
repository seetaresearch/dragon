# --------------------------------------------------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from collections import defaultdict
from dragon.__init__ import *
import dragon.protos.dragon_pb2 as pb
import dragon.config as config
from scope import GetOperatorName
from dragon.utils import MakeOperatorDef

class GraphGradientMaker(object):
    @classmethod
    def CreateGradientForOp(cls, op_def, g_output):
        """ parse ops from string """
        g_ops, g_inputs, defaults = CreateGradientDefsCC(op_def.SerializeToString(), g_output)
        for idx, g_op in enumerate(g_ops):
            new_def = pb.OperatorDef()
            new_def.ParseFromString(g_op)
            _, new_def.name = GetOperatorName()
            g_ops[idx] = new_def
        return g_ops, g_inputs, defaults

    @classmethod
    def CheckMissingGrad(cls, forward_op, inputs_to_grads, blacklist, targets):
        """ check the missing grads, if True, skip this op """
        if forward_op.type in config.NO_GRADIENT_OPERATORS:
            for input in forward_op.input: blacklist.add(input)
            return (True, None)

        # generate virtual grads for targets if necessary
        gen_grads = []
        for idx, output in enumerate(forward_op.output):
            if output not in inputs_to_grads:
                if output in targets:
                    gen_grads.append((output, idx))
                    inputs_to_grads[output] = output + '_grad'

        #  check
        for output in forward_op.output:
            if inputs_to_grads.get(output, None) is None:
                # check failed: skip backward
                if output in blacklist: return (True, gen_grads)
                if len(forward_op.output) == 1: return (True, gen_grads)

        # check pass, even if missing some grads
        return (False, gen_grads)

    @classmethod
    def Make(cls, ops, targets):
        inputs_to_grads = {}
        inputs_count = defaultdict(int)
        grads_count = defaultdict(int)
        all_g_ops = []
        all_split_grads = set()
        blacklist = set()

        # PLAY for the forward
        for op in ops:
            if op.type in config.NO_GRADIENT_OPERATORS: continue
            for input in op.input: inputs_count[input] += 1

        # PLAY for the backward
        for forward_op in ops[::-1]:
            is_skip, gen_grads = cls.CheckMissingGrad(forward_op, inputs_to_grads, blacklist, targets)
            g_outputs = list(inputs_to_grads.get(name, None) for name in forward_op.output)
            g_ops, g_inputs, defaults = cls.CreateGradientForOp(forward_op, g_outputs)

            # append ops
            if not is_skip:
                if len(gen_grads) > 0:
                    op_inputs = []; op_outputs = []; values = []
                    for item in gen_grads:
                        op_inputs.append(item[0])
                        op_outputs.append(item[0] + '_grad')
                        values.append(defaults[item[1]])
                    gen_op = MakeOperatorDef('GradientGenerate', op_inputs, op_outputs,
                                             GetOperatorName()[1], defaults=values)
                    if forward_op.HasField('device_option'):
                        gen_op.device_option.CopyFrom(forward_op.device_option)
                    all_g_ops.append(gen_op)
                for g_op in g_ops: all_g_ops.append(g_op)

            # split & gather grads for multi-used input
            for g_op in g_ops:
                for g_output_idx, g_output in enumerate(g_op.output):
                    original_idx = -1
                    for g_input_idx, g_input in enumerate(g_inputs):
                        if g_output == g_input: original_idx = g_input_idx
                    if original_idx == -1: continue
                    original_name = forward_op.input[original_idx]
                    if inputs_count[original_name] > 1:
                        # split
                        split_name = g_output + '_autosplit_%d' % grads_count[g_output]
                        if not is_skip: all_split_grads.add(split_name)
                        grads_count[g_output] += 1
                        # gather
                        if grads_count[g_output] == inputs_count[original_name]:
                            split_inputs = []
                            for idx in range(grads_count[g_output]):
                                if '%s_autosplit_%d' % (g_output, idx) in all_split_grads:
                                    split_inputs.append('%s_autosplit_%d' % (g_output, idx))
                            gather_op = MakeOperatorDef('GradientGather', split_inputs, [g_output])
                            if g_op.HasField('device_option'):
                                gather_op.device_option.CopyFrom(g_op.device_option)
                            _, gather_op.name = GetOperatorName()
                            all_g_ops.append(gather_op)
                        g_op.output[g_output_idx] = split_name

            # done
            if not is_skip:
                for name, grad in zip(forward_op.input, g_inputs):
                    if grad != '': inputs_to_grads[name] = grad
        return ops, all_g_ops