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
"""Base module class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import numpy

from dragon.core.framework import config
from dragon.core.util import string
from dragon.vm.torch.core import cpp
from dragon.vm.torch.core.autograd import grad_mode
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.tensor import Tensor
from dragon.vm.torch.core.utils import hooks


class Module(object):
    """The base class of modules.

    Inherit this class to design a new module:

    ```python
    class MyModule(torch.nn.Module):
        def __init__():
            super(MyModule, self).__init__()
    ```

    """

    class _IncompatibleKeys(collections.namedtuple(
            'IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

        def __repr__(self):
            if not self.missing_keys and not self.unexpected_keys:
                return '<All keys matched successfully>'
            return super(Module._IncompatibleKeys, self).__repr__()

        __str__ = __repr__

    def __init__(self):
        """Create a ``Module``."""
        self._modules = collections.OrderedDict()
        self._parameters = collections.OrderedDict()
        self._buffers = collections.OrderedDict()
        self._forward_hooks = collections.OrderedDict()
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

    def add_module(self, name, module):
        """Add a submodule to the module.

        Parameters
        ----------
        name : str
            The buffer name.
        module : dragon.vm.torch.nn.Module, optional
            The submodule to be registered.

        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        if hasattr(self, name) and name not in self._modules:
            raise KeyError("Attribute '{}' already exists".format(name))
        self._modules[name] = module

    def apply(self, fn):
        """Apply the function over submodules.

        Parameters
        ----------
        fn : callable
            The function to call.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def buffers(self, recurse=True):
        """Return an iterator over all buffers.

        Parameters
        ----------
        recurse : bool, optional, default=True
            Yield parameters recursively or not.

        Returns
        -------
        Iterator
            The iterator of buffer.

        """
        for name, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def children(self):
        """Return an iterator over immediate modules.

        Returns
        -------
        Iterator
            The iterator of module.

        """
        for name, module in self.named_children():
            yield module

    def cpu(self):
        """Switch the buffers and parameters to cpu device.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(lambda t: t.cpu())

    def cuda(self, device=None):
        """Switch the buffers and parameters to cuda device.

        If :attr:`device` is not provided, use the value
        set by ``dragon.cuda.set_default_device()``.

        Parameters
        ----------
        device : int, optional
            The optional device index.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        if device is None:
            device = config.config().device_index
        return self._apply(lambda t: t.cuda(device))

    def double(self):
        """Switch the buffers and parameters to ``float64``.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(lambda t: t.double_() if t.is_floating_point() else t)

    def eval(self):
        """Set to the evaluation mode.

        This method is identical to ``Module.train(False)``.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self.train(False)

    def extra_repr(self):
        """Set the extra representation."""
        return ''

    def float(self):
        """Switch the buffers and parameters to ``float32``.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(lambda t: t.float_() if t.is_floating_point() else t)

    def forward(self, *inputs, **kwargs):
        """Define the computation performed at every call.

        All subclasses should override this method:

        ```python
        class MyModule(torch.nn.Module):
            def forward(*inputs, **kwargs):
                pass
        ```

        """

    def half(self):
        """Switch the buffers and parameters to ``float16``.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(lambda t: t.half_() if t.is_floating_point() else t)

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dict from other module.

        Typically, states can only be loaded from the same module class:

        ```python
        mm = type(m)()
        mm.load_state_dict(m.state_dict())
        ```

        Set ``strict`` to ``False`` to load a mismatched dict:

        ```python
        # States matching the name successfully will be loaded
        # Otherwise, we will directly ignore them
        mm.load_state_dict(m.state_dict(), strict=False)
        ```

        Parameters
        ----------
        state_dict : dict
            The state dict.
        strict : bool, optional, default=True
            ``True`` to verify the names strictly.

        Returns
        -------
        namedtuple
            The namedtuple with ``missing_keys`` and ``unexpected_keys``.

        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        def load(module, prefix=''):
            module._load_from_state_dict(
                state_dict, prefix, True,
                missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '
                    .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '
                    .format(', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'
                .format(self.__class__.__name__, "\n\t".join(error_msgs)))

        return self._IncompatibleKeys(missing_keys, unexpected_keys)

    def modules(self):
        """Return an iterator over all modules.

        Returns
        -------
        Iterator
            The iterator of module.

        """
        for name, module in self.named_modules():
            yield module

    def mps(self, device=None):
        """Switch the buffers and parameters to mps device.

        If :attr:`device` is not provided, use the value
        set by ``dragon.mps.set_default_device()``.

        Parameters
        ----------
        device : int, optional
            The optional device index.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        if device is None:
            device = config.config().device_index
        return self._apply(lambda t: t.mps(device))

    def named_buffers(self, prefix='', recurse=True):
        """Return an iterator over all buffers.

        Parameters
        ----------
        prefix : str, optional, default=''
            The prefix added to the name.
        recurse : bool, optional, default=True
            Yield buffers recursively or not.

        Returns
        -------
        Iterator
            The iterator of (name, buffer).

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for name, buffer in gen:
            yield name, buffer

    def named_children(self):
        """Return an iterator over immediate modules, yield as ``(name, module)``.

        Returns
        -------
        Iterator
            The iterator of module.

        """
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def named_modules(self, memo=None, prefix=''):
        """Return an iterator over all modules, yield as ``(name, module)``.

        Parameters
        ----------
        memo : Set, optional
            The optional set to collect modules.
        prefix : str, optional, default=''
            The prefix added to the name.

        Returns
        -------
        Iterator
            The iterator of (name, module).

        """
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

    def named_parameters(self, prefix='', recurse=True):
        """Return an iterator over all parameters.

        Parameters
        ----------
        prefix : str, optional, default=''
            The prefix added to the name.
        recurse : bool, optional, default=True
            Yield parameters recursively or not.

        Returns
        -------
        Iterator
            The iterator of (name, param).

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for name, param in gen:
            yield name, param

    def parameters(self, recurse=True):
        """Return an iterator over all parameters.

        Parameters
        ----------
        recurse : bool, optional, default=True
            Yield parameters recursively or not.

        Returns
        -------
        Iterator
            The iterator of param.

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def register_buffer(self, name, tensor):
        """Add a buffer to the module.

        Parameters
        ----------
        name : str
            The buffer name.
        tensor : dragon.vm.torch.Tensor
            The tensor to be registered.

        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("Attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("Cannot assign '{}' object to buffer '{}'."
                            .format(type(tensor), name))
        else:
            self._buffers[name] = tensor

    def register_forward_hook(self, hook):
        """Register forward hook on the module.

        Parameters
        ----------
        hook : callable
            The hook function.

        Returns
        -------
        RemovableHandle
            The handle to remove this hook by calling ``handle.remove()``.

        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def register_parameter(self, name, param):
        """Add a parameter to the module.

        This method is identical to assign a parameter as attribute:

        ```python
        m = torch.nn.Module()
        weight = torch.nn.Parameter(torch.ones(1))
        m.register_parameter('weight', weight)  # Style1
        m.weight = weight  # Style2
        ```

        Parameters
        ----------
        name : str
            The buffer name.
        param : dragon.vm.torch.Tensor, optional
            The tensor to be registered.

        """
        if '_parameters' not in self.__dict__:
            raise AttributeError("Cannot assign parameter before init.")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("Attribute '{}' already exists.".format(name))
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("Cannot assign '{}' object to parameter '{}'."
                            .format(type(param), name))
        else:
            self._parameters[name] = param

    def state_dict(self, destination=None, prefix='', to_numpy=False):
        """Return a dict stored the buffers and parameters.

        Usually, we will use this method to renew another module:

        ```python
        m2.load_state_dict(m1.state_dict())
        ```

        Set ``to_numpy`` if you want to serialize these states:

        ```python
        # Currently, ``torch.Tensor`` is not supported to pickle
        # Convert tensors to numpy arrays before pickling
        np_states = m.state_dict(to_numpy=True)

        with open('states.pkl', 'wb') as f:
            pickle.dump(np_states, f, pickle.HIGHEST_PROTOCOL)
        ```

        Parameters
        ----------
        destination : dict, optional
            The optional output dict.
        prefix : str, optional, default=''
            The prefix added to the name of states.
        to_numpy : bool, optional, default=False
            ``True`` to store the numpy array instead.

        Returns
        -------
        Dict
            The state dict.

        """
        if destination is None:
            destination = collections.OrderedDict()
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

    def to(self, *args, **kwargs):
        """Convert states to the specified data type or device.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)
        for arg in args:
            if isinstance(arg, cpp.dtype):
                dtype = arg
            elif isinstance(arg, cpp.device):
                device = arg
            elif isinstance(arg, Tensor):
                dtype, device = arg.dtype, arg.device
                break
            else:
                raise ValueError('Unsupported conversion target.')
        if device is not None:
            if device.type == 'cpu':
                self.cpu()
            else:
                {'cuda': self.cuda,
                 'mps': self.mps}[device.type](device.index)
        if dtype is not None:
            return {'float16': self.half,
                    'float32': self.float,
                    'float64': self.double}[dtype]()
        return self

    def train(self, mode=True):
        """Set the training mode.

        Parameters
        ----------
        mode : bool, optional, default=True
            The training mode.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def zero_grad(self):
        """Set the gradient of parameters to zero."""
        for p in self.parameters():
            grad = p.grad
            if grad is not None:
                grad.zero_()

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        for param in self._parameters.values():
            if param is not None:
                with grad_mode.no_grad():
                    fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def _get_name(self):
        """Return the class name."""
        return self.__class__.__name__

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load buffers and parameters from the state dict for this module only."""
        local_name_params = itertools.chain(
            self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if input_param.shape != param.shape:
                    error_msgs.append(
                        'Size of param({}) is ({}), while load from: ({}).'
                        .format(name, ', '.join(
                            [str(d) for d in param.shape]),
                            ', '.join([str(d) for d in input_param.shape])))
                if isinstance(input_param, Tensor):
                    param.copy_(input_param)
                elif isinstance(input_param, numpy.ndarray):
                    param._impl.FromNumpy(input_param.copy(), True)
                else:
                    error_msgs.append(
                        'Excepted the input param is either '
                        'torch.Tensor or numpy.ndarray, got {}.'
                        .format(type(input_param)))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]
                    if input_name not in self._modules \
                            and input_name not in local_state:
                        unexpected_keys.append(key)

    def _named_members(self, getter, prefix='', recurse=True):
        """Return the named members."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = getter(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def __call__(self, *args, **kwargs):
        """Run the forward pipeline."""
        outputs = self.forward(*args, **kwargs)
        for hook in self._forward_hooks.values():
            hook_outputs = hook(self, args, outputs)
            if hook_outputs is not None:
                outputs = hook_outputs
        return outputs

    def __repr__(self):
        """Return a debug string."""
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = string.add_indent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def __setattr__(self, key, value):
        """Override it for detecting functions."""
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError('Cannot assign parameter before init.')
            return self.register_parameter(key, value)
        elif params is not None and key in params:
            if value is not None:
                raise TypeError("Cannot assign '{}' as parameter '{}'."
                                .format(type(value), key))
            return self.register_parameter(key, value)
        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError('Cannot assign module before init.')
            modules[key] = value
            return
        elif modules is not None and key in modules:
            if value is not None:
                raise TypeError("Cannot assign '{}' as child module '{}'."
                                .format(type(value), key))
            modules[key] = value
            return
        buffers = self.__dict__.get('_buffers')
        if buffers is not None and key in buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError("Cannot assign '{}' as buffer '{}'."
                                .format(type(value), key))
            buffers[key] = value
            return
        object.__setattr__(self, key, value)

    def __setstate__(self, state):
        """Override to restore the module dict."""
        self.__dict__.update(state)
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = collections.OrderedDict()
