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
"""Zero optimizer."""

import collections
import numpy

from dragon.core.distributed import backend as dist_backend
from dragon.core.framework import workspace
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.tensor import Tensor
from dragon.vm.torch.core.optim.optimizer import Optimizer


class ZeroReduceBucket(object):
    """Bucket structure for reducing parameters."""

    def __init__(self, params, dtype):
        self.params = params if isinstance(params, (tuple, list)) else list(params)
        self.dtype = dtype

    @property
    def size(self):
        """Return the size in bytes of managed parameters."""
        return sum([p.nbytes for p in self.params])

    @classmethod
    def new(cls, params, bucket_size=536870912):
        """Create the new buckets for given parameters."""
        coll = collections.defaultdict(list)
        buckets, dtypes = [], ("float16", "bfloat16", "float32")
        [coll[p.dtype].append(p) for p in params]
        [buckets.extend(cls.new_with_dtype(coll[t], t, bucket_size)) for t in dtypes]
        return buckets

    @classmethod
    def new_with_dtype(cls, params, dtype, bucket_size=536870912):
        """Create the new buckets along data type for given parameters."""
        buckets, cur_bucket_size = [ZeroReduceBucket([], dtype)], 0
        for p in ZeroReduceBucket(params, dtype).params[::-1]:  # Inverse order.
            next_bucket_size = cur_bucket_size + p.nbytes
            if next_bucket_size > bucket_size and len(buckets[-1].params) > 0:
                buckets.append(ZeroReduceBucket([], dtype))
                cur_bucket_size, next_bucket_size = 0, p.nbytes
            buckets[-1].params.append(p)
            cur_bucket_size = next_bucket_size
        return buckets if len(params) > 0 else []

    @staticmethod
    def get_grad(execute_ws, param, summed=False):
        """Return the grad of a parameter."""
        impl = execute_ws.get_tensor(param.id + "_grad_sum") if summed else None
        impl = execute_ws.get_tensor(param.id + "_grad") if impl is None else impl
        return Tensor(device=param.device, impl=impl) if impl else None

    def grads_and_params(self, summed=False):
        """Return the managed gradients and parameters."""
        execute_ws = workspace.get_workspace()
        grads = [self.get_grad(execute_ws, p, summed) for p in self.params]
        return grads, self.params


class ZeroOptimizer(Optimizer):
    """The base class of zero optimizers."""

    def __init__(self, params, defaults, **kwargs):
        """Create a ``ZeroOptimizer``."""
        super(ZeroOptimizer, self).__init__(params, defaults, **kwargs)
        for group in self.param_groups:
            group["buckets"] = ZeroReduceBucket.new(group["params"], self.bucket_size)

    def _update_group(self, group, params, grads):
        """Update parameters for the group."""
        dist_group = group.pop("dist_group")
        execute_ws = workspace.get_workspace()
        # Update hyper tensors with current value.
        for hyper_name in self._hyper_dict.keys():
            group_dict = self._hyper_dict[hyper_name]
            group_name, new_value = group["name"], group[hyper_name]
            if group_name not in group_dict:
                impl_name = group_name + "/" + hyper_name
                group_dict[group_name] = execute_ws.create_tensor(impl_name)
            if hyper_name == "grad_scale":
                new_value = new_value / group.pop("dist_size", 1)
            group_dict[group_name].FromNumpy(numpy.array(new_value, "float32"), False)
        # Update parameters foreach bucket.
        for bucket_idx, bucket in enumerate(group["buckets"]):
            bucket_grads, bucket_params = bucket.grads_and_params(self._sums_grad)
            extra_outputs = [group["found_inf"]] if "found_inf" in group else []
            lr_scales = [getattr(p, "lr_scale", 1) for p in bucket_params]
            Function.apply(
                self._op_type,
                bucket_params[0].device,
                bucket_grads,
                outputs=bucket_params + extra_outputs,
                name=group["name"],
                bucket_name="Bucket_%d" % (bucket_idx + 1),
                lr_scales=lr_scales,
                use_lr_scales=any([x != 1 for x in lr_scales]),
                **dist_group.arguments,
            )

    def step(self):
        """Update all parameter groups using gradients."""
        dist_group = dist_backend.get_group()
        params_all, grads_all = [], []
        for group in self.param_groups:
            group["dist_group"] = dist_group
            group["dist_size"] = dist_group.size
            grads_all.append([])
            params_all.append([])
        self._sums_grad = False
        if self._handle_custom_step:
            return params_all, grads_all
        for idx, group in enumerate(self.param_groups):
            self._update_group(group, params_all[idx], grads_all[idx])
