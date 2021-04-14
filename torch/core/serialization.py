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
#     <https://github.com/pytorch/pytorch/blob/master/torch/serialization.py>
#
# ------------------------------------------------------------
"""Serialization utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib
import sys

from dragon.core.util import six

PICKLE_MODULE = six.moves.pickle
DEFAULT_PROTOCOL = 2


def save(obj, f, pickle_module=PICKLE_MODULE, pickle_protocol=DEFAULT_PROTOCOL):
    """Save an object using pickle.

    Parameters
    ----------
    obj : Any
        The object to serialize.
    f : file_like
        The file object or file name.
    pickle_module : module, optional
        The optional pickle module.
    pickle_protocol : int, optional
        The optional pickle protocol.

    """
    return _with_file_like(
        f, 'wb', lambda f: _save(obj, f, pickle_module, pickle_protocol))


def load(f, pickle_module=PICKLE_MODULE):
    """Load an object using pickle.

    Parameters
    ----------
    f : file_like
        The file object or file name.
    pickle_module : module
        The optional pickle module.

    Returns
    -------
    Any
        The deserialized object.

    """
    try:
        return _with_file_like(
            f, 'rb', lambda f: pickle_module.load(f))
    except UnicodeDecodeError:
        return _with_file_like(
            f, 'rb', lambda f: pickle_module.load(f, encoding='bytes'))


def _save_dict(obj):
    """Recursively save the dict."""
    py_dict = type(obj)()
    for k, v in obj.items():
        if isinstance(v, dict):
            py_dict[k] = _save_dict(v)
        elif hasattr(v, 'numpy'):
            py_dict[k] = getattr(v, 'numpy')()
        else:
            py_dict[k] = v
    return py_dict


def _save(obj, f, pickle_module, pickle_protocol):
    """Pickle the object into binary file."""
    if isinstance(obj, dict):
        pickle_module.dump(_save_dict(obj), f, pickle_protocol)
    elif hasattr(obj, 'numpy'):
        pickle_module.dump(getattr(obj, 'numpy')(), f, pickle_protocol)
    else:
        pickle_module.dump(obj, f, pickle_protocol)


def _with_file_like(f, mode, body):
    """Execute a body function with a file object for f."""
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()
