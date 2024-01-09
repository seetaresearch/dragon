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
"""Builtin ops."""

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation

    ops = deprecation.not_installed("nvidia.dali")

from dragon.vm.dali.core.framework import context


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

    def __new__(cls, **kwargs):
        """Create a ``ExternalSource`` operator.

        Returns
        -------
        nvidia.dali.ops.ExternalSource
            The operator.

        """
        return ops.ExternalSource(device=context.get_device_type(), **kwargs)
