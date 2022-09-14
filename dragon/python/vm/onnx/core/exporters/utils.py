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

from dragon.core.util.registry import Registry as _Registry
from dragon.vm.onnx.core import helper

# Global registry.py to store known exporters.
_GLOBAL_REGISTERED_EXPORTERS = _Registry('exporters')
register = _GLOBAL_REGISTERED_EXPORTERS.register


class TranslatorContext(object):
    """Context to pass translator resources."""

    def __init__(
        self,
        workspace,
        blob_names,
        blob_shapes,
        blob_versions,
        opset_version,
    ):
        self.ws = workspace
        self.blob_names = blob_names
        self.blob_shapes = blob_shapes
        self.blob_versions = blob_versions
        self.opset_version = opset_version

    def unique_name(self, name):
        self.blob_versions[name] += 1
        if self.blob_versions[name] > 1:
            return name + '_%d' % (self.blob_versions[name] - 1)
        return name


def translate(op_def, context):
    """Translate the OpDef to a NodeProto.

    Parameters
    ----------
    op_def : OperatorDef
        The definition of a operator.
    context : TranslatorContext
        The context of translator.

    Returns
    -------
    NodeProto
        The node.
    Sequence[TensorProto]
        The constant tensors.

    """
    node = helper.make_node(
        op_type=op_def.type,
        inputs=op_def.input,
        outputs=op_def.output,
        name=op_def.name if op_def.name != '' else None
    )
    const_tensors = None
    return node, const_tensors


def registered_exporters():
    """Return the name of registered exporters.

    Returns
    -------
    Sequence[str]
        The exporter names.

    """
    return _GLOBAL_REGISTERED_EXPORTERS.keys


@register(['PythonPlugin', 'PythonPluginInfer'])
def python_exporter(op_def, context):
    """Export the python operators."""
    return None, None
