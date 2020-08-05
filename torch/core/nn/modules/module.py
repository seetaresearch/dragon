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
import numpy

from dragon.core.framework import config
from dragon.core.util import logging
from dragon.core.util import string
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

        If ``device`` is not provided, use the value
        set by ``dragon.config.set_cuda_device()``.

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
        """Switch the buffers and parameters to **float64**.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(
            lambda t: t.double_()
            if t.is_floating_point() else t,
        )

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
        """Switch the buffers and parameters to **float32**.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(
            lambda t: t.float_()
            if t.is_floating_point() else t,
        )

    def forward(self, *inputs, **kwargs):
        """Define the implementation of forward.

        All subclasses should override this method:

        ```python
        class MyModule(torch.nn.Module):
            def forward(*inputs, **kwargs):
                pass
        ```

        """

    def half(self):
        """Switch the buffers and parameters to **float16**.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The self.

        """
        return self._apply(
            lambda t: t.half_()
            if t.is_floating_point() else t,
        )

    def load_state_dict(self, state_dict, strict=True, verbose=False):
        """Load the state dict from other module.

        Typically, states can only loaded from the same module class:

        ```python
        mm = type(m)()
        mm.load_state_dict(m.state_dict())
        ```

        Set ``strict`` to **False** to load a mismatched dict:

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
            **True** to verify the names strictly.
        verbose : bool, optional, default=False
            **True** to print the state info.

        """
        if verbose:
            logging.info('Load the state dict.')
        unexpected = []
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                state_shape = own_state[name].shape
                param_shape = param.shape
                if state_shape != param_shape:
                    raise ValueError(
                        'Size of state({}) is ({}), while load from: ({}).'
                        .format(name, ', '.join(
                            [str(d) for d in state_shape]),
                            ', '.join([str(d) for d in param_shape])))
                if isinstance(param, Tensor):
                    own_state[name].copy_(param)
                elif isinstance(param, numpy.ndarray):
                    own_state[name]._impl.FromNumpy(param.copy())
                else:
                    raise ValueError(
                        'Excepted the type of source state is either '
                        'torch.Tensor or numpy.ndarray, got {}.'.format(type(param)))
                if verbose:
                    logging.info(
                        'Tensor({}) loaded, size: ({})'
                        .format(name, ', '.join([str(d) for d in param_shape])))
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

    def modules(self):
        """Return an iterator over all modules.

        Returns
        -------
        Iterator
            The iterator of module.

        """
        for name, module in self.named_modules():
            yield module

    def named_children(self):
        """Return an iterator over immediate modules, yield as *(name, module)*.

        Returns
        -------
        Iterator
            The iterator of module.

        """
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def named_modules(self, memo=None, prefix=''):
        """Return an iterator over all modules, yield as *(name, module)*.

        Parameters
        ----------
        memo : Set, optional
            The optional set to collect modules.
        prefix : str, optional, default=''
            The prefix added to the name.

        Returns
        -------
        Iterator
            The iterator of module.

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

    def named_parameters(self, memo=None, prefix=''):
        """Return an iterator over all parameters.

        Parameters
        ----------
        memo : Set, optional
            The optional set to collect parameters.
        prefix : str, optional, default=''
            The prefix added to the name.

        Returns
        -------
        Iterator
            The iterator of parameter.

        """
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def parameters(self):
        """Return an iterator over all parameters.

        Returns
        -------
        Iterator
            The iterator of parameter.

        """
        for name, param in self.named_parameters():
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
            raise TypeError(
                "Cannot assign '{}' object to buffer '{}' "
                "(torch Tensor or None required)"
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
            raise AttributeError("Cannot assign parameter before Module.__init__() call")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("Attribute '{}' already exists".format(name))
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                "Cannot assign '{}' object to parameter '{}' "
                "(torch.nn.Parameter or None required)"
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
            **True** to store the numpy array instead.

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
                fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def _get_name(self):
        return self.__class__.__name__

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
                raise AttributeError('Cannot assign parameters before Module.__init__().')
            self.register_parameter(key, value)
        elif params is not None and key in params:
            if value is not None:
                raise TypeError(
                    "Cannot assign '{}' as parameter '{}' "
                    "(torch.nn.Parameter or None expected)"
                    .format(type(value), key))
            self.register_parameter(key, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError('Cannot assign module before Module.__init__().')
                modules[key] = value
            elif modules is not None and key in modules:
                if value is not None:
                    raise TypeError(
                        "Cannot assign '{}' as child module '{}' "
                        "(torch.nn.Module or None expected)"
                        .format(type(value), key))
                modules[key] = value
            else:
                object.__setattr__(self, key, value)

    def __setstate__(self, state):
        """Override to restore the module dict."""
        self.__dict__.update(state)
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = collections.OrderedDict()
