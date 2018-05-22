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

from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.modules.control_flow import Copy


def _copy(dst, src):
    if id(dst) == id(src): return dst
    ctx = MakeContext(inputs=[dst])
    key = 'torch/ops/copy/{}:{}'.format(ctx[0].lower(), ctx[1])
    module = get_module(Copy, key, ctx)
    return module.forward(dst, src)