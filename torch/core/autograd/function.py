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
"""Function library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.op_lib import OpSchema
from dragon.core.framework import context
from dragon.core.framework import proto_util
from dragon.core.framework import tapes
from dragon.core.framework import workspace
from dragon.vm.torch.core.autograd import grad_mode
from dragon.vm.torch.core.tensor import Tensor


class OpExec(object):
    """The executable operator."""

    _created_instances = {}

    def __init__(self, op_type):
        self._op_type = op_type
        self._ignore_keys = {'outputs'}
        def_args = {}
        def_args_getter = OpSchema.get_args(op_type)
        if def_args_getter is not None:
            def_args = def_args_getter()
        for k, v in def_args.items():
            if k.endswith('_desc'):
                self._ignore_keys.add(k.split('_desc')[0])
        self._config_cache = {}

    @classmethod
    def get_instance(cls, op_type):
        """Return the executable operator."""
        try:
            instance = cls._created_instances[op_type]
        except KeyError:
            instance = OpExec(op_type)
            cls._created_instances[op_type] = instance
        return instance

    def get_config(self, device, **kwargs):
        """Return the execution config."""
        cache_key = self._op_type + '/' + str(device)
        for k, v in kwargs.items():
            if k not in self._ignore_keys:
                cache_key += '/' + str(v)
        try:
            return self._config_cache[cache_key]
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
                    def_args[k] = '$NAME/' + name
            op_def = proto_util.make_operator_def(
                op_type=self._op_type,
                name=kwargs.get('name', ''),
                device_option=device.to_proto(False),
                cache_key=cache_key,
                to_impl=True, **def_args)
            config = {'def': op_def,
                      'device': device,
                      'check_device': check_device,
                      'no_grad': no_grad,
                      'feed_dict': feed_dict}
            self._config_cache[cache_key] = config
            return config


class Function(object):
    """Apply registered operators."""

    @staticmethod
    def apply(op_type, device, inputs, **kwargs):
        """Apply the operator to inputs.

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
        op_exec = OpExec.get_instance(op_type)
        run_config = op_exec.get_config(device, **kwargs)
        return Function.forward(inputs, run_config, **kwargs)

    @staticmethod
    def register(op_type, args_getter=None):
        """Register an operator.

        Parameters
        ----------
        op_type : str
            The operator type.
        args_getter : callable, optional
            The callable to return the arguments.

        """
        def decorated(inner_function):
            return OpSchema.register_args(op_type, inner_function)
        if args_getter is not None:
            return OpSchema.register_args(op_type, args_getter)
        return decorated

    @staticmethod
    def forward(inputs, run_config, **kwargs):
        """Compute the function outputs."""
        graph_tape = tapes.get_tape()
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

        # Unify grad modes.
        no_grad = run_config['no_grad']
        no_grad = no_grad or not grad_mode.is_grad_enabled()
        enable_grad = enable_grad and not no_grad
        if hasattr(graph_tape, '_exporting'):
            # Ensure the intermediates saved for the exporting graph.
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
                    spec._device = device.copy()
                    outputs.append(spec)
                    outputs_id.append(spec.id)
                else:
                    outputs_id.append(spec)
                if enable_grad and outputs_id[-1] not in inputs_id:
                    raise RuntimeError('Output tensor should be in inputs if requires grad.')

        # Specialize def for given inputs and outputs.
        op_name = ''  # Optional operator name.
        op_def = run_config['def'].DeriveTo(inputs_id, outputs_id)

        # Record def if grad is enabled.
        if len(inputs) > 0 and not no_grad:
            if enable_grad:
                op_tape = tapes.OrderedTape()
                op_name = execute_ws.create_handle(op_def.type)
                op_def.name = op_name
                op_tape.add_element(op_def)
                op_tape.add_handle(op_name)
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

        # Ensure the named operator for the tracing graph.
        if hasattr(graph_tape, '_tracing'):
            if not op_name:
                op_name = execute_ws.create_handle(op_def.type)
            op_def.name = op_name
            graph_tape.add_element(op_def)
            graph_tape.add_handle(op_name)

        # Save inputs for the checkpointing graph.
        if hasattr(graph_tape, '_checkpointing'):
            for input in inputs:
                if input._tape:
                    if input._retains_grad:
                        graph_tape.add_source(input)
                elif input._requires_grad:
                    graph_tape.add_source(input)

        # Emit to dispatch this execution.
        for feed_key, value_type in run_config['feed_dict'].items():
            dest = execute_ws.create_tensor(op_name + '/' + feed_key)
            dest.FromNumpy(numpy.array(kwargs[feed_key], value_type), True)
        execute_ws.run_operator(op_def)

        # Return single or repeated outputs.
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def backward(outputs, grad_outputs, retain_graph=False):
        """Compute the function derivatives w.r.t graph leaves."""
        # Collect tapes for graph reversely.
        graph_tape = tapes.OrderedTape()
        graph_leaves, memo = set(), set()
        inputs = list(outputs)
        while len(inputs) > 0:
            input = inputs.pop(0)
            if id(input) in memo:
                continue
            memo.add(id(input))
            if input._tape:
                graph_tape.merge_from(input._tape)
                inputs.extend(input._tape.get_sources())
                input._tape = None
                if input._retains_grad:
                    graph_leaves.add(input.id)
            elif input._requires_grad:
                graph_leaves.add(input.id)

        # Emit to dispatch backward execution.
        execute_ws = workspace.get_workspace()
        execute_ws.run_backward(
            op_defs=graph_tape.get_elements(),
            targets=[y.id for y in outputs],
            grad_targets=[dy.id for dy in grad_outputs],
            sources=list(graph_leaves))

        # Free handles if graph not retained.
        if not retain_graph:
            for handle in graph_tape.get_handles():
                execute_ws.release_handle(handle)
