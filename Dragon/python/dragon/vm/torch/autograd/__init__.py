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

from dragon.vm.torch.autograd.variable import Variable
from dragon.vm.torch.autograd.grad_mode import no_grad
from dragon.vm.torch.autograd.grad_mode import enable_grad
from dragon.vm.torch.autograd.grad_mode import set_grad_enabled