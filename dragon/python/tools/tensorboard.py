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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import time
try:
    # Python 2.x
    from StringIO import StringIO as BytesIO
except ImportError:
    # Python 3.x
    from io import BytesIO

import numpy as np
import PIL.Image
try:
    import tensorflow as tf
except ImportError:
    tf = None


class TensorBoard(object):
    """The board app based on TensorFlow.

    Examples:

    ```python
    board = TensorBoard(log_dir='./logs')
    board.scalar_summary('loss', '2.3', step=0)
    board.histogram_summary('weights', np.ones((2, 3)), step=0)
    board.image_summary('images', [im], step=0)
    ```

    """

    def __init__(self, log_dir=None):
        """Create a summary writer logging to log_dir.

        If ``log_dir`` is None, ``./logs/localtime`` will be used.

        Parameters
        ----------
        log_dir : str, optional
            The root dir for monitoring.

        """
        if tf is None:
            raise ImportError('Failed to import ``tensorflow`` package.')
        if log_dir is None:
            log_dir = './logs/' + time.strftime(
                '%Y%m%d_%H%M%S', time.localtime(time.time()))
        if tf.__version__ > '2.0':
            self.writer = tf.summary.create_file_writer(log_dir)
        else:
            self.writer = tf.summary.FileWriter(log_dir)

    def close(self):
        """Close the board and apply all cached summaries."""
        self.writer.close()

    def histogram_summary(self, tag, values, step, buckets=10):
        """Write a histogram of values.

        Parameters
        ----------
        tag : str
            The key of the summary.
        values : Union[numpy.ndarray, Sequence]
            The values to be shown in the histogram.
        step : number
            The global step.
        buckets : int, optional, default=10
            The number of buckets to use.

        """
        if tf.__version__ > '2.0':
            with self.writer.as_default():
                tf.summary.histogram(tag, values, step, buckets=buckets)
        else:
            counts, bin_edges = np.histogram(values, bins=buckets)
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))
            bin_edges = bin_edges[1:]
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.writer.add_summary(summary, step)
            self.writer.flush()

    def image_summary(self, tag, images, step, order='BGR'):
        """Write a list of images.

        Parameters
        ----------
        tag : str
            The key of the summary.
        images : Union[numpy.ndarray, Sequence[numpy.ndarray]]
            The images to show.
        step : number
            The global step.
        order : {'BGR', 'RGB'}, optional
            The color order of input images.

        """
        if tf.__version__ > '2.0':
            if isinstance(images, (tuple, list)):
                images = np.stack(images)
            if len(images.shape) != 4:
                raise ValueError('Images can not be packed to (N, H, W, C).')
            if order == 'BGR':
                images = images[:, :, :, ::-1]
            with self.writer.as_default():
                tf.summary.image(tag, images, step, max_outputs=images.shape[0])
        else:
            img_summaries = []
            for i, img in enumerate(images):
                if len(img.shape) != 3:
                    raise ValueError('Excepted images in (H, W, C).')
                s = BytesIO()
                if order == 'BGR':
                    img = img[:, :, ::-1]
                PIL.Image.fromarray(img).save(s, format='png')
                img_sum = tf.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=img.shape[0],
                    width=img.shape[1],
                )
                img_summaries.append(tf.Summary.Value(
                    tag='%s/%d' % (tag, i), image=img_sum))
            self.writer.add_summary(tf.Summary(value=img_summaries), step)
            self.writer.flush()

    def scalar_summary(self, tag, value, step):
        """Write a scalar.

        Parameters
        ----------
        tag : str
            The key of the summary.
        value : scalar
            The scalar value.
        step : number
            The global step.

        """
        if tf.__version__ > '2.0':
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step)
        else:
            value = tf.Summary.Value(tag=tag, simple_value=value)
            self.writer.add_summary(tf.Summary(value=[value]), step)
            self.writer.flush()
