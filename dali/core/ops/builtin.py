# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import ops
except ImportError:
    ops = None

from dragon.vm.dali.core import context


class ExternalSource(object):
    """Create a placeholder providing data from feeding.

    Examples:

    ```python
    class MyPipeline(dali.Pipeline):
        def __init__():
            super(MyPipeline, self).__init()__
            self.image = dali.ops.ExternalSource()

        def define_graph(self):
            self.image = self.image()

        def iter_setup(self):
            images = [np.ones((224, 224, 3), 'uint8')]
            self.feed_input(self.image, images)
    ```

    """

    def __new__(cls):
        """Create a ``ExternalSource`` operator.

        Returns
        -------
        nvidia.dali.ops.ExternalSource
            The operator.

        """
        return ops.ExternalSource(device=context.get_device_type())
