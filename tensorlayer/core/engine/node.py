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

from dragon.core.util import nest


class LayerNode(object):
    """A conceptional node wrapping for a layer.

    Parameters
    ----------
    layer : tl.layers.Layer
        A tl layer that wants to create a node.
    node_index : int
        Index of this node in layer._nodes.
    in_nodes ：a list of LayerNode
        Father nodes to this node.
    in_tensors : a list of tensors
        Input tensors to this node.
    out_tensors : a list of tensors
        Output tensors to this node.
    in_tensor_idxes : a list of int
        Indexes of each input tensor in its corresponding node's out_tensors.

    """

    def __init__(
        self,
        layer,
        node_index,
        in_nodes,
        in_tensors,
        out_tensors,
        in_tensor_idxes,
    ):
        self.layer = layer
        self.node_index = node_index
        self.in_nodes = in_nodes
        self.out_nodes = []
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.name = layer.name + "_node_{}".format(node_index)
        self.in_tensors_idxes = in_tensor_idxes
        self.visited = False

    def __call__(self, inputs, **kwargs):
        outputs = self.layer.forward(inputs, **kwargs)
        self.in_tensors = nest.flatten(inputs)
        self.out_tensors = nest.flatten(outputs)
        return self.out_tensors
