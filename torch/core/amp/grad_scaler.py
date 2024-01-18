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
"""Gradient scaler."""

from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.tensor import Tensor


class GradScaler(object):
    """Scale model outputs to control the gradient magnitude."""

    def __init__(
        self,
        init_scale=16384,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    ):
        """Create a ``GradScaler``.

        Parameters
        ----------
        init_scale : number, optional, default=16384
            The initial scale factor.
        growth_factor : float, optional, default=2.0
            The multiplier to scale for growth.
        backoff_factor : float, optional, default=0.5
            The multiplier to scale for backoff.
        growth_interval : int, optional, default=2000
            The interval to grow the scale factor.
        enabled : bool, optional, default=True
            Enable the scaling or not.

        """
        self._enabled = enabled
        self._scale, self._scale_value = Tensor([init_scale]).squeeze_(), None
        self._found_inf, self._found_inf_value = Tensor(0), None
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0

    def get_backoff_factor(self):
        """Return the scale backoff factor.

        Returns
        -------
        float
            The scale backoff factor.

        """
        return self._backoff_factor

    def get_growth_factor(self):
        """Return the scale growth factor.

        Returns
        -------
        float
            The scale growth factor.

        """
        return self._growth_factor

    def get_growth_interval(self):
        """Return the growth_interval.

        Returns
        -------
        int
            The growth interval.

        """
        return self._growth_interval

    def get_scale(self):
        """Return the current scale factor.

        Returns
        -------
        float
            The current scale factor.

        """
        if self._enabled:
            if self._scale_value is None:
                self._scale_value = float(self._scale)
            return self._scale_value
        return 1.0

    def is_enabled(self):
        """Return a bool indicating if scaling is enabled.

        Returns
        -------
        bool
            ``True`` if enabled else ``False``.

        """
        return self._enabled

    def scale(self, outputs):
        """Apply scale to the outputs.

        Parameters
        ----------
        outputs : Union[dragon.vm.torch.Tensor, Iterable[dragon.vm.torch.Tensor]]
            The outputs to scale.

        Returns
        -------
        Union[dragon.vm.torch.Tensor, Iterable[dragon.vm.torch.Tensor]]
            The scaled outputs.

        """

        def apply_scale(x):
            dtype = x.dtype if x.dtype != self._scale.dtype else None
            device = x.device if x.device != self._scale.device else None
            return x.mul_(self._scale.to(device=device, dtype=dtype))

        if self._enabled and isinstance(outputs, Tensor):
            return apply_scale(outputs)
        if self._enabled and isinstance(outputs, (tuple, list)):
            return [apply_scale(x) for x in outputs]
        return outputs

    def set_backoff_factor(self, new_factor):
        """Set the scale backoff factor.

        Parameters
        ----------
        new_factor : float
            The new scale backoff factor.

        """
        self._backoff_factor = new_factor

    def set_growth_factor(self, new_factor):
        """Set the scale growth factor.

        Parameters
        ----------
        new_factor : float
            The new scale growth factor.

        """
        self._growth_factor = new_factor

    def set_growth_interval(self, new_interval):
        """Set the scale growth factor.

        Parameters
        ----------
        new_interval : int
            The new growth interval.

        """
        self._growth_interval = new_interval

    def step(self, optimizer):
        """Step the given optimizer.

        Parameters
        ----------
        optimizer : dragon.vm.torch.optim.Optimizer
            The optimizer.

        """
        if not self._enabled:
            return optimizer.step()
        optimizer._handle_custom_step = True
        params_all, grads_all = optimizer.step()
        optimizer._handle_custom_step = False
        grads = sum(grads_all, [])
        Function.apply("GradientCheck", grads[0].device, grads, outputs=[self._found_inf])
        self._found_inf_value = float(self._found_inf)
        if self._found_inf_value:
            return
        for idx, group in enumerate(optimizer.param_groups):
            group["grad_scale"] = 1.0 / self.get_scale()
            optimizer._update_group(group, params_all[idx], grads_all[idx])

    def update(self, new_scale=None):
        """Update the scale factor.

        Parameters
        ----------
        new_scale : Union[number, dragon.vm.torch.Tensor], optional
            The new scale factor.

        """
        if not self._enabled:
            return
        if new_scale is None:
            if self._found_inf_value:
                self._growth_tracker = 0
                new_scale = self.get_scale() * self._backoff_factor
            else:
                self._growth_tracker += 1
                if self._growth_tracker == self._growth_interval:
                    new_scale = self.get_scale() * self._growth_factor
                    self._growth_tracker = 0
        if isinstance(new_scale, (float, int)) and new_scale != self._scale_value:
            self._scale.fill_(float(new_scale))
            self._scale_value = float(self._scale)
        elif isinstance(new_scale, Tensor):
            self._scale.copy_(new_scale)
            self._scale_value = float(self._scale)
