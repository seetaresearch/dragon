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

from collections import OrderedDict

import numpy as np
import dragon as dg
from dragon.core.scope import get_tensor_scope
import dragon.core.utils as pb_utils
from dragon.config import logger

from dragon.vm.torch.environ import \
    add_submodule, get_module_name

from dragon.vm.torch.tensor import Tensor, Parameter
from dragon.vm.torch.tensor_pool import TPool
from dragon.vm.torch.execute_engine import RunOperator


class Module(object):
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._persistent_key = self._op = None
        self._ctx = ('CPU', 0)

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
        """Override it for detecting modules.

        Returns
        -------
        None

        """
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

    def _load_state_dict_key_mismatch(self, full_name, name, is_missing):
        pass

    def load_state_dict(self, state_dict, strict=True, verbose=True):
        if verbose: logger.info('Load the state dict.')
        def submodule_key_mismatch(full_name, is_missing):
            module = self
            names = full_name.split(".")
            for module_name in names[:-1]:
                if module_name in module._modules:
                    module = module._modules[module_name]
                else:
                    return
            module._load_state_dict_key_mismatch(full_name, names[-1], is_missing)

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
                if own_state[name].dtype != str(param.dtype):
                    raise ValueError('DType of state({}) is {}, \n'
                        'While load from a PyArray of {}.'.format(name,
                        own_state[name].dtype, str(param.dtype)))
                if isinstance(param, Tensor):
                    own_state[name].copy_(param)
                elif isinstance(param, np.ndarray):
                    dg.tensor_utils.SetPyArray(own_state[name], param)
                else:
                    raise ValueError('Excepted the type of source state is either '
                        'torch.Tensor or numpy.ndarray, got {}.'.format(type(param)))
                if verbose:
                    logger.info('* Tensor({}) loaded, Size: ({})'.format(name,
                            ', '.join([str(d) for d in param_shape])))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            # pass the mismatch info to submodules so that they have a chance to
            # raise a custom class-specific error
            for name in unexpected:
                submodule_key_mismatch(name, False)
            for name in missing:
                submodule_key_mismatch(name, True)
            error_msg = ''
            if len(unexpected) > 0:
                error_msg += 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected))
            if len(missing) > 0:
                error_msg += 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing))
            if len(error_msg) > 0:
                raise KeyError(error_msg)

    def register_buffer(self, name, tensor):
        """Adds a buffer to the module.

        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(type(tensor), name))
        else:
            self._buffers[name] = tensor

    def register_parameter(self, name, param):
        """Adds a parameter to the module.

        """
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
        with dg.name_scope(get_module_name(self)):
            return self.forward(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError('The base module can not be called.')

    def name_scope(self, remove_separator=True):
        scope = get_tensor_scope()
        if remove_separator and scope[-1] == '/': scope = scope[:-1]
        return scope

    def register_buffers(self, n_buffers):
        """Apply for n buffers from TensorPool.

        Buffers will be released after backward pass.

        Parameters
        ----------
        n_buffers : int
            The number of buffers.

        """
        return [TPool.get() for i in range(n_buffers)]

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
        """Returns an iterator over module parameters.

        """
        if memo is None: memo = set()
        # this module
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        # sub modules
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def parameters(self):
        """Returns an iterator over module parameters.

        """
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

    def cpu(self):
        self._ctx = ('CPU', 0)
        # Remove key & op to re-create a one with new ctx
        self._persistent_key = self._op = None
        return self._apply(lambda t: t.cpu(),
                           lambda m: m.cpu())

    def cuda(self, device=None):
        if device is None: device = dg.config.GetGPU()
        self._ctx = ('CUDA', device)
        # Remove key & op to re-create a one with new ctx
        self._persistent_key = self._op = None
        return self._apply(lambda t: t.cuda(device),
                           lambda m: m.cuda(device))

    def _gen_persistent_key(self):
        self._persistent_key = '{}{}:{}'.format(
            self.name_scope(False), self._ctx[0].lower(), self._ctx[1])

    @property
    def persistent_key(self):
        if self._persistent_key is None:
            self._gen_persistent_key()
        return self._persistent_key

    def _gen_op(self):
        self._op = pb_utils.MakeOperatorDef(op_type=self.op_meta['op_type'], name='runtime',
            inputs=['I({})'.format(i) for i in range(self.op_meta['n_inputs'])],
            outputs=['O({})'.format(i) for i in range(self.op_meta['n_outputs'])],
            device_option=pb_utils.MakeDeviceOption({'CPU': 0, 'CUDA': 1}[self._ctx[0]],
                                                     self._ctx[1], engine='CUDNN'),
            persistent_key=self.persistent_key, **self.op_meta['arguments'])
        dg.workspace.CreatePersistentOp(self._op)
        return self._op

    @property
    def op(self):
        if self._op is None: self._gen_op()
        return self._op

    def register_op(self):
        raise NotImplementedError()

    def register_output(self, dtype='float32'):
        return (dtype, self._ctx)

    def unify_devices(self, inputs):
        for ix, t in enumerate(inputs):
            if t._ctx[0] != self._ctx[0] or \
                t._ctx[1] != self._ctx[1]:
                    raise ValueError('Module({}) is defined at {}:{}, '
                        '\nFound Input({}) is at {}:{}.'.format(
                            self.name_scope(True), self._ctx[0], self._ctx[1],
                            ix, t._ctx[0], t._ctx[1]))

    def run(self, inputs, outputs, auto_grad=True):
        meta = ('PERSISTENT', self.persistent_key, self.op)
        return RunOperator(inputs, outputs, meta, auto_grad=auto_grad)