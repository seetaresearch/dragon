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

from dragon.vm.tensorflow.framework import ops
from dragon.vm.tensorflow.ops import var_scope as variable_scope


def get_variables(
    scope=None,
    suffix=None,
    collection=ops.GraphKeys.GLOBAL_VARIABLES,
):
    if isinstance(scope, variable_scope.VariableScope):
        scope = scope.name
    if suffix is not None:
        scope = (scope or '') + '.*' + suffix
    return ops.get_collection(collection, scope)