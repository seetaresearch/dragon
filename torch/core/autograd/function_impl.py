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
"""Function implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.op_impl import OpSchema
from dragon.core.autograph import tape
from dragon.core.framework import context
from dragon.core.framework import proto_util
from dragon.core.framework import workspace
from dragon.vm.torch.core.autograd import grad_mode
from dragon.vm.torch.core.tensor import Tensor


class ExecutionCache(object):
    """Container of cached executions."""

    _created_instances = {}

    def __init__(self, op_type):
        self._op_type = op_type
        self._ignore_keys = {'outputs', 'name'}
        def_args = {}
        def_args_getter = OpSchema.get_args(op_type)
        if def_args_getter is not None:
            def_args = def_args_getter()
        for k, v in def_args.items():
            if k.endswith('_desc'):
                self._ignore_keys.add(k.split('_desc')[0])
        self._cache_dict = {}

    @classmethod
    def get_cache(cls, op_type):
        """Return the cache of given operator type."""
        try:
            instance = ExecutionCache._created_instances[op_type]
        except KeyError:
            instance = ExecutionCache(op_type)
            ExecutionCache._created_instances[op_type] = instance
        return instance

    def get_config(self, device, **kwargs):
        """Return the config from given arguments."""
        cache_key = self._op_type + '/' + str(device)
        for k, v in kwargs.items():
            if k not in self._ignore_keys:
                cache_key += '/' + str(v)
        try:
            return self._cache_dict[cache_key]
        except KeyError:
            def_args, feed_dict = {}, {}
            def_args_getter = OpSchema.get_args(self._op_type)
            if def_args_getter is not None:
                def_args = def_args_getter(**kwargs)
            device = def_args.pop('device', device)
            check_device = def_args.pop('check_device', True)
            no_grad = def_args.pop('no_grad', False)
            for k, v in def_args.items():
                if k.endswith('_desc') and v:
                    name = k.split('_desc')[0]
                    feed_dict[name] = v
                    def_args[k] = '$HANDLE/' + name
            op_def = proto_util.make_operator_def(
                op_type=self._op_type,
                name=kwargs.get('handle', ''),
                device_option=device.to_proto(False),
                cache_key=cache_key,
                to_impl=True, **def_args)
            cache_value = {'def': op_def,
                           'device': device,
                           'check_device': check_device,
                           'no_grad': no_grad,
                           'feed_dict': feed_dict}
            self._cache_dict[cache_key] = cache_value
            return cache_value


class FunctionLib(object):
    """Library to apply functions via registered operators."""

    @staticmethod
    def apply(op_type, device, inputs, **kwargs):
        """Apply a function to inputs.

        Parameters
        ----------
        op_type : str
            The operator type.
        device : dragon.vm.torch.device
            The execute device.
        inputs : Sequence[dragon.vm.torch.Tensor]
            The input tensors.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output tensors.

        """
        cache = ExecutionCache.get_cache(op_type)
        run_config = cache.get_config(device, **kwargs)
        return FunctionLib._forward(inputs, run_config, **kwargs)

    @staticmethod
    def register_args(op_type, args_getter=None):
        """Register arguments for an operator type.

        Parameters
        ----------
        op_type : str
            The operator type.
        args_getter : callable, optional
            The callable to return the operators arguments.

        """
        def decorated(inner_function):
            return OpSchema.register_args(op_type, inner_function)
        if args_getter is not None:
            return OpSchema.register_args(op_type, args_getter)
        return decorated

    @staticmethod
    def _forward(inputs, run_config, **kwargs):
        """Compute the function outputs."""
        graph_tape = tape.get_tape()
        execute_ws = workspace.get_workspace()
        device = run_config['device']

        # Add inputs.
        inputs_id = []
        enable_grad = False
        for i, input in enumerate(inputs):
            inputs_id.append(input.id)
            if input.requires_grad:
                enable_grad = True
            if run_config['check_device'] and input._device != device:
                raise RuntimeError(
                    'Mismatched device between function and '
                    'element {} of input tensors. ({} vs. {})'
                    .format(i, device, input._device))

        # Compute gradient flags.
        no_grad = run_config['no_grad']
        no_grad = no_grad or not grad_mode.is_grad_enabled()
        enable_grad = enable_grad and not no_grad
        if isinstance(graph_tape, tape.GraphTape):
            no_grad, enable_grad = False, True

        # Add outputs.
        outputs, outputs_id = [], []
        output_specs = kwargs.get('outputs', [None])
        for i, spec in enumerate(output_specs):
            if spec is None:
                outputs.append(Tensor(
                    device=device.copy(),
                    impl=execute_ws.create_tensor(
                        scope=context.get_variable_scope(enable_grad)),
                    deleter=execute_ws._handle_pool))
                outputs_id.append(outputs[i].id)
            else:
                if isinstance(spec, Tensor):
                    outputs.append(spec)
                    outputs_id.append(spec.id)
                else:
                    outputs_id.append(spec)
                if enable_grad and outputs_id[-1] not in inputs_id:
                    raise RuntimeError('Output tensor should be in inputs if requires grad.')

        # Specialize def for given inputs and outputs.
        op_handle = ''  # Optional handle
        op_def = run_config['def'].DeriveTo(inputs_id, outputs_id)

        # Record def if grad is enabled.
        if len(inputs) > 0 and not no_grad:
            if enable_grad:
                op_tape = tape.OrderedTape()
                op_handle = execute_ws._handle_pool.create(op_def.type)
                op_def.name = op_handle
                op_tape.add_op_def(op_def)
                for input in inputs:
                    op_tape.add_source(input)
                for output in outputs:
                    op_tape.merge_from(output._tape)
                for output in outputs:
                    output._tape = op_tape
                    output._requires_grad = True
            else:
                for output in outputs:
                    output._requires_grad = False

        # Ensure handle created for the graph.
        if isinstance(graph_tape, tape.GraphTape):
            if not op_handle:
                op_handle = execute_ws._handle_pool.create(op_def.type)
                op_def.name = op_handle
            graph_tape.add_op_def(op_def)

        # Emit to dispatch this execution.
        for feed_key, value_type in run_config['feed_dict'].items():
            dest = execute_ws.create_tensor(op_handle + '/' + feed_key)
            dest.FromNumpy(numpy.array(kwargs[feed_key], value_type), True)
        execute_ws.run_operator(op_def)

        # Return single or repeated outputs.
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def _backward(outputs, grad_outputs, retain_graph=False):
        """Compute the function derivatives w.r.t graph leaves."""
        # Collect forward tapes.
        inputs = list(outputs)
        op_tape = tape.OrderedTape()
        graph_leaves = set()
        memo = set()
        while len(inputs) > 0:
            input = inputs.pop(0)
            if id(input) in memo:
                continue
            memo.add(id(input))
            if input._tape:
                op_tape.merge_from(input._tape)
                inputs.extend(input._tape.get_sources())
                input._tape = None
                if input._retains_grad:
                    graph_leaves.add(input.id)
            elif input._requires_grad:
                graph_leaves.add(input.id)

        # Run backward computations reversely.
        op_defs = op_tape.get_op_defs()
        execute_ws = workspace.get_workspace()
        execute_ws.run_backward(
            op_defs=op_defs,
            targets=[y.id for y in outputs],
            grad_targets=[dy.id for dy in grad_outputs],
            sources=list(graph_leaves),
        )

        # Free the forward handles if allowed.
        if not retain_graph:
            handle_pool = execute_ws._handle_pool
            for op_def in op_defs:
                handle_pool.release(op_def.name)
