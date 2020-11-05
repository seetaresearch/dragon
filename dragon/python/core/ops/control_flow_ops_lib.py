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
"""Control flow ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Assign(Operator):
    """Assign operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Assign, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Assign',
            'arguments': {
                'starts_descs': [
                    '${{HANDLE}}/starts[{}]'
                    .format(n) for n in range(self.ndim)],
                'sizes_descs': [
                    '${{HANDLE}}/sizes[{}]'
                    .format(n) for n in range(self.ndim)],
            },
        }

    def feed(self, ws, handle, starts, sizes):
        """
        Add the arguments.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/starts[{}]'.format(handle, i),
                starts[i], 'int64')
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')

    def forward(self, inputs, starts, sizes):
        """
        Parameters ---------- inputs : list of outputs )

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        return self.dispatch(
            [inputs[1]], [inputs[0]],
            callback=lambda ws, handle:
                self.feed(ws, handle, starts, sizes),
            no_grad=True,
        )


class Copy(Operator):
    """Copy operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Copy, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'Copy', 'arguments': {}}

    def forward(self, inputs, outputs):
        """
        Parameters ---------- inputs : list of outputs

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            outputs: (todo): write your description
        """
        outputs = outputs if outputs else [self.alloc()]
        return self.dispatch(inputs, outputs, no_grad=True)


class MaskedAssign(Operator):
    """MaskedAssign operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(MaskedAssign, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'MaskedAssign', 'arguments': {}}

    def forward(self, inputs):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs[1:], [inputs[0]], no_grad=True)
