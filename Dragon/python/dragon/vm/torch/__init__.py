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

# Import Dynamic Methods
import dragon.vm.torch.ops.tensor

# Import Core Methods
from dragon.vm.torch.tensor import *
from dragon.vm.torch.c_api import Size, from_numpy
from dragon.vm.torch.serialization import save, load

# Import Subpackages
import dragon.vm.torch.cuda
from dragon.vm.torch.ops.builtin import *
from dragon.vm.torch.autograd import *
import dragon.vm.torch.nn
import dragon.vm.torch.optim
import dragon.vm.torch.onnx