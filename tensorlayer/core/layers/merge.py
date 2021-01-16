# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
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

from dragon.core.ops import array_ops
from dragon.core.ops import math_ops
from dragon.vm.tensorlayer.core.engine import layer
from dragon.vm.tensorlayer.core.layers import utils


class Concat(layer.Layer):
    """Layer to concat tensors according to the given axis."""

    def __init__(self, concat_dim=-1, name=None):
        """Create a ``Concat`` layer.

        Parameters
        ----------
        concat_dim : int, optional, default=-1
            The dimension to concatenate.
        name : str, optional
            The layer name.

        """
        super(Concat, self).__init__(name)
        self.concat_dim = concat_dim

    def __repr__(self):
        s = '{classname}(concat_dim={concat_dim})'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        return array_ops.concat(
            inputs,
            axis=self.concat_dim,
            name=self.name,
        )


class Elementwise(layer.Layer):
    """Layer to combine inputs by applying element-wise operation."""

    def __init__(self, operation='add', act=None, name=None):
        """Create a ``Elementwise`` layer.

        Parameters
        ----------
        operation : str, optional, default='add'
            The operation to perform.
        act : callable, optional
            The optional activation function.
        name : str, optional
            The layer name.

        """
        super(Elementwise, self).__init__(name, act)
        self.combine_fn = {
            'add': math_ops.add,
            'sub': math_ops.sub,
            'mul': math_ops.mul,
            'div': math_ops.div,
            'subtract': math_ops.sub,
            'multiply': math_ops.mul,
            'divide': math_ops.div,
            'max': math_ops.maximum,
            'min': math_ops.minimum,
            'maximum': math_ops.maximum,
            'minimum': math_ops.minimum,
        }[operation.lower()]

    def __repr__(self):
        s = '{classname}(combine_fn={combine_fn}, '
        s += utils.get_act_str(self.act)
        if self.name is not None:
            s += ', name=\'{_name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        outputs = inputs[0]
        for input in inputs[1:]:
            outputs = self.combine_fn([outputs, input])
        if self.act:
            outputs = self.act(outputs)
        return outputs
