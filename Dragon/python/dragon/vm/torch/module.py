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
#      <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py>
#
# ------------------------------------------------------------

""" A pseudo ``Module`` to be compatible with the namespace based backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import dragon
import warnings
from collections import OrderedDict

from dragon import config as _cfg
from dragon.core import scope as _scope
from dragon.core import logging as _logging
from dragon.core import proto_utils as _proto_utils
from dragon.core import tensor_utils as _tensor_utils

from dragon.vm.torch.c_api import device as _Device
from dragon.vm.torch.tensor import Tensor, Parameter
from dragon.vm.torch.execution import RunOperator
from dragon.vm.torch.environ import add_submodule, get_module_name


class Module(object):
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._device = _Device()
        self._module_key = None
        self._module_def = None
        self.training = True

    def __getattr__(self, item):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if item in _parameters:
                return _parameters[item]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if item in _buffers:
                return _buffers[item]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if item in modules:
                return modules[item]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, item))

    def __setattr__(self, key, value):
        """Override it for detecting modules."""
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            self.register_parameter(key, value)
        elif params is not None and key in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(type(value), key))
            self.register_parameter(key, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                modules[key] = value
                add_submodule(value, key)
            elif modules is not None and key in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(type(value), key))
                modules[key] = value
            else:
                object.__setattr__(self, key, value)

    def state_dict(self, destination=None, prefix='', to_numpy=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = \
                    param.cpu().numpy().copy() if to_numpy else param
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = \
                    buf.cpu().numpy().copy() if to_numpy else buf
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', to_numpy=to_numpy)
        return destination

    def load_state_dict(self, state_dict, strict=True, verbose=True):
        if verbose: _logging.info('Load the state dict.')
        unexpected = []
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                state_shape = own_state[name].shape
                param_shape = param.shape
                if state_shape != param_shape:
                    raise ValueError('Size of state({}) is ({}), \n'
                        'While load from Size of ({}).'.format(name,
                        ', '.join([str(d) for d in state_shape]),
                        ', '.join([str(d) for d in param_shape])))
                if isinstance(param, Tensor):
                    own_state[name].copy_(param)
                elif isinstance(param, numpy.ndarray):
                    _tensor_utils.SetArray(own_state[name], param)
                else:
                    raise ValueError('Excepted the type of source state is either '
                        'dragon.vm.torch.Tensor or numpy.ndarray, got {}.'.format(type(param)))
                if verbose:
                    _logging.info('Tensor({}) loaded, Size: ({})'.format(name,
                            ', '.join([str(d) for d in param_shape])))
            else:
                unexpected.append(name)
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            error_msg = ''
            if len(unexpected) > 0:
                error_msg += 'Unexpected key(s) in state_dict: {}.\n'.format(
                    ', '.join('"{}"'.format(k) for k in unexpected))
            if len(missing) > 0:
                error_msg += 'Missing key(s) in state_dict: {}.'.format(
                    ', '.join('"{}"'.format(k) for k in missing))
            if len(error_msg) > 0:
                raise KeyError(error_msg)

    def register_buffer(self, name, tensor):
        """Adds a buffer to the module."""
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(type(tensor), name))
        else:
            self._buffers[name] = tensor

    def register_parameter(self, name, param):
        """Adds a parameter to the module."""
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(type(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Variable to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another variable, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def add_module(self, name, module, name_v2=None):
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        if hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        self._modules[name] = module
        add_submodule(module, name_v2 if name_v2 else name)

    def __call__(self, *args, **kwargs):
        with dragon.name_scope(get_module_name(self)):
            return self.forward(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError('The base module can not be called.')

    def name_scope(self, remove_separator=True):
        scope = _scope.get_default_name_scope()
        if remove_separator and \
            len(scope) > 0 and \
                scope[-1] == '/':
                    scope = scope[:-1]
        return scope

    def children(self):
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def named_parameters(self, memo=None, prefix=''):
        """Returns an iterator over module parameters."""
        if memo is None: memo = set()
        # This module
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

        # Sub modules
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def parameters(self):
        """Returns an iterator over module parameters."""
        for name, param in self.named_parameters():
            yield param

    def _apply(self, p_fn, m_fn=None):
        for module in self.children():
            if m_fn: m_fn(module)
        for param in self._parameters.values():
            if param is not None: p_fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = p_fn(buf)
        return self

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cpu(self):
        self._device = _Device()
        # Remove key and op to re-create a one with new device
        self._module_key = self._module_def = None
        return self._apply(lambda t: t.cpu(),
                           lambda m: m.cpu())

    def cuda(self, device=None):
        if device is None: device = dragon.config.GetGPU()
        self._device = _Device('cuda', device)
        # Remove key and op to re-create a one with new device
        self._module_key = self._module_def = None
        return self._apply(lambda t: t.cuda(device),
                           lambda m: m.cuda(device))

    def float(self):
        return self._apply(
            lambda t: t.float_() if t.is_floating_point() else t,
                lambda m: m.float())

    def double(self):
        return self._apply(
            lambda t: t.double_() if t.is_floating_point() else t,
                lambda m: m.double())

    def half(self):
        return self._apply(
            lambda t: t.half_() if t.is_floating_point() else t,
                lambda m: m.half())

    def _gen_module_key(self):
        self._module_key = '{}{}'.format(
            self.name_scope(False), self._device)

    @property
    def module_key(self):
        if self._module_key is None:
            self._gen_module_key()
        return self._module_key

    def _gen_module_def(self):
        rng_seed = _cfg.GetGlobalOptions()['random_seed']
        self._module_def = \
            _proto_utils.MakeCXXOperatorDef(
                name='runtime',
                uid=self.module_key,
                op_type=self.op_meta['op_type'],
                device_option=_proto_utils.
                    GetDeviceOption(
                        self._device.type,
                        self._device.index,
                        rng_seed=rng_seed,
                ),
                **self.op_meta['arguments']
            )

    def register_op(self):
        pass

    def register_output(self):
        return self._device.copy()

    def unify_devices(self, inputs):
        for ix, t in enumerate(inputs):
            if t._device != self._device:
                raise ValueError('Module({}) is defined at {}, '
                    '\nFound Input({}) is at {}.'.format(
                        self.name_scope(True),
                            self._device, ix, t._device))

    def run(self, inputs, outputs, auto_grad=True, callback=None):
        if self._module_def is None: self._gen_module_def()
        return RunOperator(
            inputs=inputs,
            outputs=outputs,
            meta=(self.module_key, self._module_def),
            auto_grad=auto_grad,
            callback_on_run=callback,
        )

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        warnings.warn('Module.zero_grad() is deprecated. '
            'Use ``torch.optim.Optimizer.zero_grad()`` instead.')

    def extra_repr(self):
        """Set the extra representation of the module
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

        """
        return ''

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s