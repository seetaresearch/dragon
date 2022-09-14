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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.vm.tensorflow.core.ops import variables


class Module(object):
    """The base class of neural network modules.

    Inherit this class to design a new module:

    ```python
    class MyModule(tf.Module):
        def __init__(name=None):
            super(MyModule, self).__init__(name)
    ```

    """

    # A blacklist to ignore some properties to flatten.
    _MODULE_IGNORED_PROPERTIES = frozenset(())

    def __init__(self, name=None):
        self._name = name

    @property
    def name(self):
        """Return the module name.

        Returns
        -------
        str
            The module name.

        """
        if self._name is None:
            self._init_set_name()
        return self._name

    @property
    def name_scope(self):
        """Returns a ``dragon.name_scope`` instance for this class.

        Returns
        -------
        ContextManger
            The context manager to apply the name scope.

        """
        return context.name_scope(self.name)

    @property
    def submodules(self):
        """Return all the submodules into a sequence.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Module]
            The submodules.

        """
        return tuple(self._flatten(predicate=_is_module))

    @property
    def trainable_variables(self):
        """Return all the trainable variables into a sequence.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The trainable variables.

        """
        return tuple(self._flatten(predicate=_is_trainable_variable))

    @property
    def variables(self):
        """Return all the variables into a sequence.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The variables.

        """
        return tuple(self._flatten(predicate=_is_variable))

    def _flatten(
        self,
        recursive=True,
        predicate=None,
        attribute_traversal_key=None,
        with_path=False,
    ):
        """Flatten this module to select attributes.

        Parameters
        ----------
        recursive : bool, optional, default=True
            ``True`` to traverse the submodules recursively.
        predicate : callable, optional
            The callable to select attribute.
        attribute_traversal_key : callable, optional
            The custom key function to be used in ``sorted(...)``.
        with_path : bool, optional, default=True
            ``True`` to return *(paths, element)* otherwise *element*.

        Returns
        -------
        Iterator
            The iterator of attributes.

        """
        return flatten_module(
            self,
            recursive=recursive,
            predicate=predicate if predicate is not None else (lambda _: True),
            attributes_to_ignore=self._MODULE_IGNORED_PROPERTIES,
            attribute_traversal_key=attribute_traversal_key,
            with_path=with_path,
        )

    def _init_set_name(self, name=None, zero_based=True):
        if name is None:
            self._name = workspace.get_workspace().unique_name(
                name=camel_to_snake(self.__class__.__name__),
                namespace='Object',
                zero_based=zero_based)
        else:
            if not valid_identifier(name):
                raise ValueError('<name> should be a legal identifier.')
            self._name = name


def camel_to_snake(value):
    """Convert the name from camel-style to snake-style."""
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', value)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    if insecure[0] != '_':
        return insecure
    return insecure[2:]


def flatten_module(
    module,
    recursive,
    predicate,
    attribute_traversal_key,
    attributes_to_ignore,
    with_path,
    module_path=(),
    seen=None,
):
    """Flatten attributes according to the predicate."""
    if seen is None:
        seen = {id(module)}
    module_dict = vars(module)
    submodules = []
    for key in sorted(module_dict, key=attribute_traversal_key):
        if key in attributes_to_ignore:
            continue
        for leaf_path, leaf in nest.flatten_with_paths(module_dict[key]):
            leaf_path = (key,) + leaf_path
            if not with_path:
                leaf_id = id(leaf)
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)
            if predicate(leaf):
                if with_path:
                    yield module_path + leaf_path, leaf
                else:
                    yield leaf
            if recursive and _is_module(leaf):
                submodules.append((module_path + leaf_path, leaf))
    for submodule_path, submodule in submodules:
        subvalues = flatten_module(
            submodule,
            recursive=recursive,
            predicate=predicate,
            attribute_traversal_key=attribute_traversal_key,
            attributes_to_ignore=submodule._MODULE_IGNORED_PROPERTIES,
            with_path=with_path,
            module_path=submodule_path,
            seen=seen,
        )
        for subvalue in subvalues:
            yield subvalue


def valid_identifier(name):
    """Return if the name can be a python identifier."""
    return bool(_VALID_IDENTIFIER.match(name))


def _is_module(obj):
    """Return if the object is a instance of module."""
    return isinstance(obj, Module)


def _is_variable(obj):
    """Return if the object is a variable."""
    return isinstance(obj, variables.Variable)


def _is_trainable_variable(obj):
    """Return if the object is a trainable variable."""
    return _is_variable(obj) and getattr(obj, "trainable", False)


_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_]([a-zA-Z0-9_])*$")
