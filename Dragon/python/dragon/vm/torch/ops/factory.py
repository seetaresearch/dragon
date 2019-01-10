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


_GLOBAL_TORCH_BUILTIN_MODULES = {}


def has_module(key):
    return key in _GLOBAL_TORCH_BUILTIN_MODULES


def register_module(cls, key, ctx, **kwargs):
    global _GLOBAL_TORCH_BUILTIN_MODULES
    module = cls(key, ctx, **kwargs)
    _GLOBAL_TORCH_BUILTIN_MODULES[key] = module
    return module


def get_module(cls, key, ctx, **kwargs):
    if has_module(key): return _GLOBAL_TORCH_BUILTIN_MODULES[key]
    return register_module(cls, key, ctx, **kwargs)