# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/pytorch/pytorch/blob/master/torch/serialization.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import pathlib
import sys

from dragon.core.util import six


PICKLE_MODULE = six.moves.pickle
DEFAULT_PROTOCOL = PICKLE_MODULE.HIGHEST_PROTOCOL


def _is_real_file(f):
    """Check if f is backed by a real file."""
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False


def _with_file_like(f, mode, body):
    """Execute a body function with a file object for f."""
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        dir = os.path.dirname(f)
        # Bug fix: empty directory, i.e., under the work directory.
        if dir != '' and not os.path.exists(dir):
            os.makedirs(dir)
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()


def _save_dict(obj):
    """Recursively save the dict."""
    py_dict = type(obj)()
    for k, v in obj.items():
        if isinstance(v, dict):
            py_dict[k] = _save_dict(v)
        elif hasattr(v, 'numpy'):
            py_dict[k] = v.numpy()
        else:
            py_dict[k] = v
    return py_dict


def _save(obj, f, pickle_module, pickle_protocol):
    """Pickle the object into binary file."""
    if not isinstance(obj, dict):
        raise ValueError('Currently only the state dict can be saved.')
    py_dict = type(obj)()
    for k, v in obj.items():
        if isinstance(v, dict):
            py_dict[k] = _save_dict(v)
        elif hasattr(v, 'numpy'):
            py_dict[k] = v.numpy()
        else:
            py_dict[k] = v
    pickle_module.dump(py_dict, f, pickle_protocol)


def save(obj, f, pickle_module=PICKLE_MODULE, pickle_protocol=DEFAULT_PROTOCOL):
    return _with_file_like(
        f, "wb", lambda f: _save(obj, f, pickle_module, pickle_protocol)
    )


def _load(f, map_location=None, pickle_module=six.moves.pickle, file=None):
    try:
        return pickle_module.load(f)
    except UnicodeDecodeError:
        if file:
            # ReOpen the file, because the MARK is corrupted.
            f = open(file, 'rb')
            return pickle_module.load(f, encoding='iso-8859-1')
        else:
            return pickle_module.load(f, encoding='iso-8859-1')


def load(f, map_location=None, pickle_module=PICKLE_MODULE):
    new_fd = False
    file = None
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        file = f
        f = open(f, 'rb')
    try:
        return _load(f, map_location, pickle_module, file)
    finally:
        if new_fd:
            f.close()
