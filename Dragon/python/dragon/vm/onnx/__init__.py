# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.onnx.utils import (
    # Make a value info for easy exporting
    make_value_info,
    # Perform a surgery on GraphDef for easy exporting
    surgery_on_graph_def,
    # Export a ONNX model from the GraphDef
    export_from_graph_def,
    # Export a ONNX model from the text format of GraphDef
    export_from_graph_text,
    # Import the ONNX model to a GraphDef
    import_to_graph_def,
    # Import a ONNX model to a Function
    import_to_function,
)