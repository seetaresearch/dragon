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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import pipeline
    from dragon.vm.dali.core.framework import context

    class Pipeline(pipeline.Pipeline):
        """The base pipeline class to define operations.

        ```python
        class MyPipeline(dali.Pipeline):
            def __init__(batch_size=1, num_threads=4):
                super(MyPipeline, self).__init__(batch_size, num_threads)
        ```

        """

        def __init__(
            self,
            batch_size=1,
            num_threads=1,
            seed=3,
            prefetch_queue_depth=2,
            py_num_workers=1,
            **kwargs
        ):
            """Create a ``Pipeline``.

            Parameters
            ----------
            batch_size : int, optional, default=1
                The number of examples in a batch.
            num_threads : int, optional, default=1
                The number of threads to execute the operations.
            seed : int, optional, default=3
                The seed for random generator.
            prefetch_queue_depth : int, optional, default=2
                The number of prefetch queues.
            py_num_workers : int, optional, default=1
                The number of workers to process external source.

            """
            device = context.get_device()
            if device['device_type'] == 'cpu':
                device['device_index'] = None
            super(Pipeline, self).__init__(
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device['device_index'],
                seed=seed,
                prefetch_queue_depth=prefetch_queue_depth,
                py_num_workers=py_num_workers,
                **kwargs
            )

        @property
        def batch_size(self):
            """Return the batch size of pipeline.

            Returns
            -------
            int
                The batch size.

            """
            return self._max_batch_size

        @property
        def device_id(self):
            """Return the device index of pipeline.

            Returns
            -------
            int
                The device index.

            """
            return self._device_id

        @property
        def max_batch_size(self):
            """Return the maximum batch size of pipeline.

            Returns
            -------
            int
                The maximum batch size.

            """
            return self._max_batch_size

        @property
        def num_threads(self):
            """Return the number of threads to execute pipeline.

            Returns
            -------
            int
                The number of threads.

            """
            return self._num_threads

        def build(self, define_graph=None):
            """Build the pipeline.

            Parameters
            ----------
            define_graph : callable, optional
                The defined function to use instead.

            """
            super(Pipeline, self).build(define_graph)

        def define_graph(self):
            """Define the symbolic operations for pipeline."""
            super(Pipeline, self).define_graph()

        def feed_input(self, *args, **kwargs):
            """Bind an array to the edge reference."""
            super(Pipeline, self).feed_input(*args, **kwargs)

except ImportError:

    class Pipeline(object):
        """The base pipeline class to define operations.

        ```python
        class MyPipeline(dali.Pipeline):
            def __init__(batch_size=1, num_threads=4):
                super(MyPipeline, self).__init__(batch_size, num_threads)
        ```

        """

        def __init__(
            self,
            batch_size=1,
            num_threads=1,
            seed=3,
            prefetch_queue_depth=2,
            py_num_workers=1,
            **kwargs
        ):
            """Create a ``Pipeline``

            Parameters
            ----------
            batch_size : int, optional, default=1
                The number of examples in a batch.
            num_threads : int, optional, default=1
                The number of threads to execute the operations.
            seed : int, optional, default=3
                The seed for random generator.
            prefetch_queue_depth : int, optional, default=2
                The number of prefetch queues.
            py_num_workers : int, optional, default=1
                The number of workers to process external source.

            """
            self._max_batch_size = batch_size
            self._num_threads = num_threads
            self._seed = seed
            self._prefetch_queue_depth = prefetch_queue_depth

        @property
        def batch_size(self):
            """Return the batch size of pipeline.

            Returns
            -------
            int
                The batch size.

            """
            return self._max_batch_size

        @property
        def device_id(self):
            """Return the device index of pipeline.

            Returns
            -------
            int
                The device index.

            """
            return 0

        @property
        def max_batch_size(self):
            """Return the maximum batch size of pipeline.

            Returns
            -------
            int
                The maximum batch size.

            """
            return self._max_batch_size

        @property
        def num_threads(self):
            """Return the number of threads to execute pipeline.

            Returns
            -------
            int
                The number of threads.

            """
            return self._num_threads

        def build(self, define_graph=None):
            """Build the pipeline.

            Parameters
            ----------
            define_graph : callable, optional
                The defined function to use instead.

            """
            pass

        def define_graph(self):
            """Define the symbolic operations for pipeline."""
            pass

        def feed_input(self, *args, **kwargs):
            """Bind an array to the edge reference."""
            pass
