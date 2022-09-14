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
"""Generic utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import inspect
from dragon.core.util import six


def deserialize_keras_object(
    identifier,
    module_objects,
    printable_module_name='object',
):
    """Deserialize the keras object."""
    if isinstance(identifier, six.string_types):
        object_name = identifier
        obj = module_objects.get(object_name)
        if obj is None:
            raise ValueError(
                'Unknown ' + printable_module_name + ': ' + object_name)
        if inspect.isclass(obj):
            return obj()
        return obj
    elif inspect.isfunction(identifier):
        return identifier
    else:
        raise TypeError(
            'Could not interpret the {} identifier: {}.'
            .format(printable_module_name, identifier))
