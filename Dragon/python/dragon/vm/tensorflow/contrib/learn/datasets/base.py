# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py>
#
# ------------------------------------------------------------

"""Base utilities for loading datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import time

import shutil
from six.moves import urllib

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
    """Simple decorator for wrapping retriable functions.

    Args:
      initial_delay: the initial delay.
      factor: each subsequent retry, the delay is multiplied by this value.
          (must be >= 1).
      jitter: to avoid lockstep, the returned delay is multiplied by a random
          number between (1-jitter) and (1+jitter). To add a 20% jitter, set
          jitter = 0.2. Must be < 1.
      max_delay: the maximum delay allowed (actual max is
          max_delay * (1 + jitter).
      is_retriable: (optional) a function that takes an Exception as an argument
          and returns true if retry should be applied.
    """
    if factor < 1:
        raise ValueError('factor must be >= 1; was %f' % (factor,))

    if jitter >= 1:
        raise ValueError('jitter must be < 1; was %f' % (jitter,))

    # Generator to compute the individual delays
    def delays():
        delay = initial_delay
        while delay <= max_delay:
            yield delay * random.uniform(1 - jitter, 1 + jitter)
            delay *= factor

    def wrap(fn):
        """Wrapper function factory invoked by decorator magic."""

        def wrapped_fn(*args, **kwargs):
            """The actual wrapper function that applies the retry logic."""
            for delay in delays():
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except)
                    if is_retriable is None:
                        continue

                    if is_retriable(e):
                        time.sleep(delay)
                    else:
                        raise
            return fn(*args, **kwargs)

        return wrapped_fn

    return wrap


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
    return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
    return urllib.request.urlretrieve(url, filename)


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url)
        shutil.copy(temp_file_name, filepath)
        size = os.path.getsize(filepath)
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
