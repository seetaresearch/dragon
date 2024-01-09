# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Generic utilities."""

from dragon.core.util import inspect


def deserialize_keras_object(
    identifier,
    module_objects,
    printable_module_name="object",
):
    """Deserialize the keras object."""
    if isinstance(identifier, str):
        object_name = identifier
        obj = module_objects.get(object_name)
        if obj is None:
            raise ValueError("Unknown " + printable_module_name + ": " + object_name)
        if inspect.isclass(obj):
            return obj()
        return obj
    elif inspect.isfunction(identifier):
        return identifier
    else:
        raise TypeError(
            "Could not interpret the {} identifier: {}.".format(printable_module_name, identifier)
        )
