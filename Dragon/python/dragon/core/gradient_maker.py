# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/caffe2/caffe2/blob/master/caffe2/python/core.py>
#
# ------------------------------------------------------------

"""A Python-Implemented gradient maker.

The basic idea of ``GradientMaker`` comes from ``caffe2``,
Jia provided a simple way to bridge the Generator(Python) with OpScheme(C++).

We also implemented a efficient C++ version for dynamic gradient flows, see,

    <https://github.com/seetaresearch/Dragon/blob/master/Dragon/src/core/graph_gradient.cc>

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from dragon.import_c_apis import *
import dragon.config as config
import dragon.protos.dragon_pb2 as pb
from dragon.core.utils import MakeOperatorDef

from .scope import GetOperatorName


def _op_name(given=None):
    if given: return given
    _, name = GetOperatorName()
    return name


class GraphGradientMaker(object):
    """
    GraphGradientMaker is deigned to generate gradient operators automatically.

    It relies on the generating rules defined in the C++ backend.

    """
    @classmethod
    def CreateGrad(cls, forward_op, g_outputs):
        """Generate the OperatorDef for ``BackwardOp`` by ``ForwardOp``.

        Parameters
        ----------
        forward_op : dragon_pb2.OperatorDef
            The OperatorDef of ``ForwardOp``.
        g_outputs : list of str or list of None
            The inputs of ``BackwardOp`` (Precomputed grads).
        name : str or None
            The optional operator name.

        Returns
        -------
        tuple
            The OpDef, outputs and defaults of ``BackwardOp``.

        References
        ----------
        The wrapper of ``CreateGradientDefsCC``.

        """
        g_ops, g_inputs, defaults = \
            CreateGradientDefsCC(forward_op.SerializeToString(), g_outputs)
        for idx, g_op in enumerate(g_ops):
            new_def = pb.OperatorDef()
            new_def.ParseFromString(g_op)
            g_ops[idx] = new_def
        return g_ops, g_inputs, defaults

    @classmethod
    def CheckGrad(cls, forward_op, inputs_to_grads, blacklist, targets):
        """Check if missing Grads. If True, skip this Op.

        Parameters
        ----------
        forward_op : dragon_pb2.OperatorDef
            The OperatorDef of ``ForwardOp``.
        inputs_to_grads : dict
            The dict of <input, g_input>.
        blacklist : set of str
            The set of ``NoGradient`` tensors.
        targets : list of str
            The solving targets.

        Returns
        -------
        tuple
            The result of checking and generated filling grads.

        """
        if forward_op.type in config.NO_GRADIENT_OPERATORS:
            for input in forward_op.input: blacklist.add(input)
            return (True, None)

        # Generate virtual grads for targets if necessary
        gen_grads = []
        for idx, output in enumerate(forward_op.output):
            if output not in inputs_to_grads:
                if output in targets:
                    gen_grads.append((output, idx))
                    inputs_to_grads[output] = output + '_grad'

        #  Check
        for output in forward_op.output:
            if inputs_to_grads.get(output, None) is None:
                # check failed: skip backward
                if output in blacklist: return (True, gen_grads)
                if len(forward_op.output) == 1: return (True, gen_grads)

        # Pass, even if missing some grads
        return (False, gen_grads)

    @classmethod
    def Make(cls, forward_ops, targets, input_grads=None, auto_names=True):
        """Make ``BackwardOps`` based on ``ForwardOps``.

        Parameters
        ----------
        forward_ops : list of dragon_pb2.OperatorDef
            The operators of ``ForwardOp``.
        targets : list of str
            The solving targets.
        input_grads : dict or None
            The input grads.
        auto_names : boolean
            Whether to use auto names for backward ops.

        Returns
        -------
        tuple
            The ``ForwardOps`` and ``BackwardOps``.

        See Also
        --------
        `theano.function(*args, **kwargs)`_ - How to make a graph. [**Theano Style**]

        """
        inputs_to_grads = {} if input_grads is None else input_grads
        inputs_count = defaultdict(int); grads_count = defaultdict(int)
        all_split_grads = set(); blacklist = set()

        backward_ops = []

        # A DAG may not have any in-place operators
        is_dag = True

        # PLAY for the forward
        for forward_op in forward_ops:
            if forward_op.type in config.NO_GRADIENT_OPERATORS: continue
            outputs = [o for o in forward_op.output]
            for input in forward_op.input:
                if input not in outputs:
                    # Avoid to count the duplicate input(i.e. the in-place output)
                    inputs_count[input] += 1
                else:
                    is_dag = False

        # PLAY for the backward
        for forward_op in forward_ops[::-1]:
            # Collect inputs & outputs
            is_skip, gen_grads = \
                cls.CheckGrad(forward_op, inputs_to_grads, blacklist, targets)
            # Missing grads are represented as ``None``
            g_outputs = list(inputs_to_grads.get(name, None) for name in forward_op.output)
            g_ops, g_inputs, defaults = cls.CreateGrad(forward_op, g_outputs)

            # Append ops
            if not is_skip:
                # --> GenOp
                if len(gen_grads) > 0:
                    op_inputs = []; op_outputs = []; values = []
                    for item in gen_grads:
                        op_inputs.append(item[0])
                        op_outputs.append(item[0] + '_grad')
                        values.append(defaults[item[1]])
                    gen_op = MakeOperatorDef('GradientGenerate', op_inputs, op_outputs,
                        name=_op_name(None if auto_names else 'runtime'), defaults=values)
                    if forward_op.HasField('device_option'):
                        gen_op.device_option.CopyFrom(forward_op.device_option)
                    backward_ops.append(gen_op)
                # --> GradOp
                for g_op in g_ops:
                    g_op.name = _op_name(None if auto_names else 'runtime')
                    backward_ops.append(g_op)

            # Split & Gather grads for multi-used input
            for g_op in g_ops:
                for g_output_idx, g_output in enumerate(g_op.output):
                    original_idx = -1
                    for g_input_idx, g_input in enumerate(g_inputs):
                        if g_output == g_input: original_idx = g_input_idx
                    # Ignore un-used && in-placed GI(?)
                    if original_idx == -1: continue
                    if g_output in g_op.input: continue
                    # Found a split branch
                    original_name = forward_op.input[original_idx]
                    if inputs_count[original_name] > 1:
                        # Split
                        split_name = g_output + '_autosplit_%d' % grads_count[g_output]
                        if not is_skip: all_split_grads.add(split_name)
                        grads_count[g_output] += 1
                        # Gather
                        if grads_count[g_output] == inputs_count[original_name]:
                            split_inputs = []
                            for idx in range(grads_count[g_output]):
                                if '%s_autosplit_%d' % (g_output, idx) in all_split_grads:
                                    split_inputs.append('%s_autosplit_%d' % (g_output, idx))
                            gather_op = MakeOperatorDef('GradientGather', split_inputs, [g_output])
                            if g_op.HasField('device_option'):
                                gather_op.device_option.CopyFrom(g_op.device_option)
                            gather_op.name = _op_name(None if auto_names else 'runtime')
                            backward_ops.append(gather_op)
                        g_op.output[g_output_idx] = split_name

            # Done
            if not is_skip:
                for name, grad in zip(forward_op.input, g_inputs):
                    if grad != '': inputs_to_grads[name] = grad

        return forward_ops, backward_ops, is_dag