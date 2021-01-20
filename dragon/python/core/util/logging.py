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
"""Logging utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import logging as _logging
import os
import sys as _sys
import threading

from dragon.core.framework import backend
from dragon.core.framework import config

_logger = None
_logger_lock = threading.Lock()


def get_logger():
    """Return the current logger."""
    global _logger
    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger
    _logger_lock.acquire()
    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('dragon')
        logger.setLevel('INFO')
        logger.propagate = False
        if True:
            # Determine whether we are in an interactive environment.
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive
            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel('INFO')
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr
            # Add the output handler.
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter('%(levelname)s %(message)s'))
            logger.addHandler(_handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()


def log(level, msg, *args, **kwargs):
    """Log message at the given level.

    Parameters
    ----------
    level : Union[int,str]
        The logging level value.
    msg: str
        The message.

    """
    level = _logging._checkLevel(level)
    get_logger().log(level, _detailed_msg(msg), *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Log message at the DEBUG level.

    Parameters
    ----------
    msg: str
        The message.

    """
    get_logger().debug(_detailed_msg(msg), *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log the message at the ERROR level.

    Parameters
    ----------
    msg: str
        The message.

    """
    get_logger().error(_detailed_msg(msg), *args, **kwargs)


def fatal(msg, *args, **kwargs):
    """Log message at the FATAL level.

    Parameters
    ----------
    msg: str
        The message.

    """
    get_logger().fatal(_detailed_msg(msg), *args, **kwargs)


def get_verbosity():
    """Return the current logging level.

    Returns
    -------
    int
        The logging level value.

    """
    return get_logger().getEffectiveLevel()


def info(msg, *args, **kwargs):
    """Log message at the INFO level.

    Parameters
    ----------
    msg: str
        The message.

    """
    get_logger().info(_detailed_msg(msg), *args, **kwargs)


def set_directory(path):
    """Set the directory for logging files.

    Parameters
    ----------
    path : str, optional
        The path of the directory.

    """
    config.config().log_dir = path


def set_verbosity(level):
    r"""Set the logging level.

    Following levels are defined (default='INFO'):

    * level = ``DEBUG``

    * level = ``INFO``

    * level = ``WARNING``

    * level = ``ERROR``

    * level = ``FATAL``

    Parameters
    ----------
    level : Union[int, str]
        The logging level value.

    """
    get_logger().setLevel(level)
    backend.SetLoggingLevel(level)


def warning(msg, *args, **kwargs):
    """Log message at the WARNING level.

    Parameters
    ----------
    msg: str
        The message.

    """
    get_logger().warning(_detailed_msg(msg), *args, **kwargs)


def _detailed_msg(msg):
    """Return the formatted message with file and lineno."""
    file, lineno = inspect.stack()[:3][2][1:3]
    return "{}:{}] {}".format(os.path.split(file)[-1], lineno, msg)
