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
"""DataType API."""

from dragon.vm.tensorflow.core.framework.dtypes import DType
from dragon.vm.tensorflow.core.framework.dtypes import as_dtype
from dragon.vm.tensorflow.core.framework.dtypes import bfloat16
from dragon.vm.tensorflow.core.framework.dtypes import bool
from dragon.vm.tensorflow.core.framework.dtypes import complex128
from dragon.vm.tensorflow.core.framework.dtypes import complex64
from dragon.vm.tensorflow.core.framework.dtypes import double
from dragon.vm.tensorflow.core.framework.dtypes import float16
from dragon.vm.tensorflow.core.framework.dtypes import float32
from dragon.vm.tensorflow.core.framework.dtypes import float64
from dragon.vm.tensorflow.core.framework.dtypes import half
from dragon.vm.tensorflow.core.framework.dtypes import int16
from dragon.vm.tensorflow.core.framework.dtypes import int32
from dragon.vm.tensorflow.core.framework.dtypes import int64
from dragon.vm.tensorflow.core.framework.dtypes import int8
from dragon.vm.tensorflow.core.framework.dtypes import qint16
from dragon.vm.tensorflow.core.framework.dtypes import qint32
from dragon.vm.tensorflow.core.framework.dtypes import qint8
from dragon.vm.tensorflow.core.framework.dtypes import quint16
from dragon.vm.tensorflow.core.framework.dtypes import quint8
from dragon.vm.tensorflow.core.framework.dtypes import string
from dragon.vm.tensorflow.core.framework.dtypes import uint16
from dragon.vm.tensorflow.core.framework.dtypes import uint32
from dragon.vm.tensorflow.core.framework.dtypes import uint64
from dragon.vm.tensorflow.core.framework.dtypes import uint8
from dragon.vm.tensorflow.core.framework.dtypes import variant
from dragon.vm.tensorflow.core.ops.math_ops import cast
