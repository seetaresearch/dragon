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


__BUILTIN_MODULES = {}


def has_module(key):
    return key in __BUILTIN_MODULES


def register_module(cls, key, ctx, **kwargs):
    global __BUILTIN_MODULES
    __BUILTIN_MODULES[key] = cls(key, ctx, **kwargs)


def get_module(cls, key, ctx, **kwargs):
    if has_module(key): return __BUILTIN_MODULES[key]
    register_module(cls, key, ctx, **kwargs)
    return __BUILTIN_MODULES[key]