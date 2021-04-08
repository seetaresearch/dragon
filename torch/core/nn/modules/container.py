# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/container.py>
#
# ------------------------------------------------------------
"""Container modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import operator
import warnings

from dragon.core.util import six
from dragon.vm.torch.core.nn.modules.module import Module


class Container(Module):
    """The base container."""

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    """The sequential module container."""

    def __init__(self, *args):
        """Create a ``Sequential`` container.

        Parameters
        ----------
        args : dragon.vm.torch.nn.Module...
            The initial modules.

        """
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for i, module in enumerate(args):
                self.add_module(str(i), module)

    def forward(self, input):
        """Call modules sequentially.

        Parameters
        ----------
        input : Any
            The container input.

        """
        for module in self._modules.values():
            input = module(input)
        return input

    def _get_item_by_index(self, iterator, index):
        """Get the idx-th item of the iterator."""
        size = len(self)
        i = operator.index(index)
        if not -size <= i < size:
            raise IndexError('Index {} is out of range'.format(i))
        i %= size
        return next(itertools.islice(iterator, i, None))

    def __delitem__(self, item):
        if isinstance(item, slice):
            keys = [key for key in list(self._modules.keys())[item]]
            for key in keys:
                del self._modules[key]
        else:
            key = self._get_item_by_index(self._modules.keys(), item)
            del self._modules[key]

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Sequential(collections.OrderedDict(
                list(self._modules.items())[item]))
        else:
            return self._get_item_by_index(self._modules.values(), item)

    def __len__(self):
        return len(self._modules)

    def __setitem__(self, key, value):
        key = self._get_item_by_index(self._modules.keys(), key)
        return setattr(self, key, value)


class ModuleList(Module):
    """The list module container."""

    def __init__(self, modules=None):
        """Create a ``ModuleList`` container.

        Parameters
        ----------
        modules : Sequence[dragon.vm.torch.nn.Module], optional
            The initial modules.

        """
        super(ModuleList, self).__init__()
        if modules is not None:
            self.__iadd__(modules)

    def append(self, module):
        """Add a module in this container.

        Parameters
        ----------
        module : dragon.vm.torch.nn.Module
            The module to add.

        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        """Add a sequence of modules in this container.

        Parameters
        ----------
        modules : Sequence[dragon.vm.torch.nn.Module], optional
            The modules to add.

        """
        if not isinstance(modules, six.collections_abc.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "Sequence, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def insert(self, index, module):
        """Add a module at a given index in this container.

        Parameters
        ----------
        index : int
            The insert index.
        module : dragon.vm.torch.nn.Module
            The module to add.

        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def _get_abs_string_index(self, index):
        """Return the absolute index for the list of functions"""
        i = operator.index(index)
        if not (-len(self) <= i < len(self)):
            raise IndexError('Index {} is out of range'.format(i))
        if i < 0:
            i += len(self)
        return str(i)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ModuleList(list(self._modules.values())[item])
        else:
            return self._modules[self._get_abs_string_index(item)]

    def __iadd__(self, modules):
        return self.extend(modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __setitem__(self, key, value):
        return setattr(self, self._get_abs_string_index(key), value)

    def __delitem__(self, item):
        if isinstance(item, slice):
            keys = [key for key in list(self._modules.keys())[item]]
            for key in keys:
                del self._modules[key]
        else:
            del self._modules[self._get_abs_string_index(item)]
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = collections.OrderedDict(
            list(zip(str_indices, self._modules.values())))
