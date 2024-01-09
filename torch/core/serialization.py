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
"""Serialization utilities."""

import pathlib
import pickle

PICKLE_MODULE = pickle
DEFAULT_PROTOCOL = 2


def save(
    obj, f, pickle_module=PICKLE_MODULE, pickle_protocol=DEFAULT_PROTOCOL, map_location="numpy"
):
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
    map_location : str, optional, default="numpy"
        The storage to save tensor data.

    """
    return _with_file_like(
        f, "wb", lambda f: _save(obj, f, pickle_module, pickle_protocol, map_location)
    )


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
        return _with_file_like(f, "rb", lambda f: pickle_module.load(f))
    except UnicodeDecodeError:
        return _with_file_like(f, "rb", lambda f: pickle_module.load(f, encoding="bytes"))


def _save_dict(obj, map_location):
    """Recursively save the dict."""
    py_dict = type(obj)()
    for k, v in obj.items():
        if isinstance(v, dict):
            py_dict[k] = _save_dict(v, map_location)
        elif hasattr(v, "numpy") and str(map_location) == "numpy":
            py_dict[k] = getattr(v, "numpy")()
        else:
            if hasattr(v, "device") and map_location is not None:
                v.__dict__["_save_device"] = map_location
            py_dict[k] = v
    return py_dict


def _save(obj, f, pickle_module, pickle_protocol, map_location):
    """Pickle the object into binary file."""
    if isinstance(obj, dict):
        pickle_module.dump(_save_dict(obj, map_location), f, pickle_protocol)
    elif hasattr(obj, "numpy") and str(map_location) == "numpy":
        pickle_module.dump(getattr(obj, "numpy")(), f, pickle_protocol)
    else:
        if hasattr(obj, "device") and map_location is not None:
            obj.__dict__["_save_device"] = map_location
        pickle_module.dump(obj, f, pickle_protocol)


def _with_file_like(f, mode, body):
    """Execute a body function with a file object for f."""
    new_fd = False
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()
