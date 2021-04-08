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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/context.py>
#
# ------------------------------------------------------------
"""Autograph context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading


class _ThreadLocalData(threading.local):
    """Thread local storage for the context."""

    def __init__(self):
        super(_ThreadLocalData, self).__init__()
        self.mode = 'EAGER_MODE'
        self.is_eager = self.mode == 'EAGER_MODE'


class Context(object):
    """Context to control the auto graph behaviors."""

    def __init__(self):
        self._thread_local_data = _ThreadLocalData()

    def executing_eagerly(self):
        """Return True if current thread has eager executing enabled."""
        return self._thread_local_data.is_eager

    @contextlib.contextmanager
    def mode(self, mode):
        """Context-manager to allow setting the mode to EAGER/GRAPH."""
        ctx = self._thread_local_data
        old_mode = ctx.mode
        old_is_eager = ctx.is_eager
        ctx.mode = mode
        ctx.is_eager = mode == 'EAGER_MODE'
        try:
            yield
        finally:
            ctx.mode = old_mode
            ctx.is_eager = old_is_eager


_context = None
_context_lock = threading.Lock()


def _initialize_context():
    global _context
    with _context_lock:
        if _context is None:
            _context = Context()


def context():
    """Return a singleton context object."""
    if _context is None:
        _initialize_context()
    return _context


def context_safe():
    """Return the current context."""
    return _context


def eager_mode():
    """Context-manager set the eager execution mode."""
    return context().mode('EAGER_MODE')


def executing_eagerly():
    """Return if the current thread has enabled eager execution."""
    return context().executing_eagerly()


def graph_mode():
    """Context-manager to set the graph execution mode."""
    return context().mode('GRAPH_M0DE')
