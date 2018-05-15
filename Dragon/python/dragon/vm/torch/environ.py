# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

# A dict for storing the global modules
__GLOBAL_MODULES = {}

# A integer for storing the uid for current global module
__GLOBAL_UID_BY_TYPE = defaultdict(int)

# A dict for storing the submodules
__SUB_MODULES = {}

# A dict for mapping buffers to module
__BUFFERS_TO_MODULE = {}


def add_submodule(module, key):
    global __SUB_MODULES
    __SUB_MODULES[id(module)] = key


def get_module_name(module):
    module_id = id(module)
    if module_id in __SUB_MODULES:
        # SubModule get name from ``self.xyz``
        return __SUB_MODULES[module_id]
    else:
        # GlobalModule get a auto name as ``g_module_type_{%d}``
        if module_id not in __GLOBAL_MODULES:
            # Create a new auto name
            global __GLOBAL_UID_BY_TYPE
            model_type = module.__class__.__name__.lower()
            __GLOBAL_MODULES[module_id] = '[auto]' + model_type + '_{}'\
                .format(__GLOBAL_UID_BY_TYPE[model_type] + 1)
            __GLOBAL_UID_BY_TYPE[model_type] += 1
        # Use a existing auto name
        return __GLOBAL_MODULES[module_id]


def is_global_module(module):
    global __GLOBAL_MODULES
    return id(module) in __GLOBAL_MODULES