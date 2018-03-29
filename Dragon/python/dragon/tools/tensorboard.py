# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>

# Code referenced from:
#
#      https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
#
# ------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import PIL.Image
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x
try:
    import tensorflow as tf
except ImportError as e:
    logging.warning(
        'Cannot import tensorflow. Error: {0}'.format(str(e)))


class TensorBoard(object):
    """The board app based on TensorFlow.

    Examples
    --------
    >>> board = TensorBoard(log_dir='./logs')
    >>> board.scalar_summary('loss', '2.3', step=0)
    >>> board.histogram_summary('weights', np.ones((2, 3)), step=0)
    >>> board.image_summary('images', [im], step=0)

    """
    def __init__(self, log_dir=None):
        """Create a summary writer logging to log_dir.

        If ``log_dir`` is None, ``./logs/localtime`` will be used.

        Parameters
        ----------
        log_dir : str or None
            The root dir for monitoring.

        Returns
        -------
        TensorBoard
            The board app.

        """
        if log_dir is None:
            log_dir = './logs/' + time.strftime('%Y%m%d_%H%M%S',
                    time.localtime(time.time()))
        self.writer = tf.summary.FileWriter(log_dir)

    def close(self):
        """Close the board and apply all cached summaries.

        Returns
        -------
        None

        """
        self.writer.close()

    def scalar_summary(self, tag, value, step):
        """Write a scalar variable.

        Parameters
        ----------
        tag : str
            The key of the summary.
        value : scalar
            The scalar value.
        step : number
            The global step.

        Returns
        -------
        None

        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step, order='BGR'):
        """Write a list of images.

        The images could be stacked in the type of ``numpy.ndarray``.

        Otherwise, the type of images should be list.

        Parameters
        ----------
        tag : str
            The key of the summary.
        images : list or numpy.ndarray
            The images to show.
        step : number
            The global step.
        order : str
            The color order. ``BGR`` or ``RGB``.

        Returns
        -------
        None

        """
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            if order == 'BGR':
                if len(img.shape) == 3: img = img[:, :, ::-1]
                elif len(img.shape) == 4: img = img[:, :, ::-1, :]

            PIL.Image.fromarray(img).save(s, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histogram_summary(self, tag, values, step, bins=1000):
        """Write a histogram of values.

        Parameters
        ----------
        tag : str
            The key of the summary.
        values : list, tuple or numpy.ndarray
            The values to be shown in the histogram.
        step : number
            The global step.
        bins : int
            The number of bins in the the histogram.

        Returns
        -------
        None

        """
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()