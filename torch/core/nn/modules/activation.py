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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops.init import functional as init_funcs
from dragon.vm.torch.core.tensor import Tensor


class ELU(Module):
    r"""Apply the exponential linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \text{ELU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    m = torch.nn.ELU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.elu(...)`_ - Apply the exponential linear unit to input.

    """

    def __init__(self, alpha=1., inplace=False):
        r"""Create a ``ELU`` module.

        Parameters
        ----------
        alpha : float, optional, default=1.
            The value to :math:`\alpha`.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)


class GumbelSoftmax(Module):
    r"""Apply the gumbel softmax function.
    `[Jang et.al, 2016] <https://arxiv.org/abs/1611.01144>`_.

    The **GumbelSoftmax** function is defined as:

    .. math::
        \text{GumbelSoftmax}(x) =
            \frac{exp((\log(\pi_{i}) + g_{i}) / \tau)}
            {\sum exp((\log(\pi_{j}) + g_{i}) / \tau)} \\
        \quad \\ \text{where}\quad g_{i} \sim \text{Gumbel}(0, 1)

    Examples:

    ```python
    m = torch.nn.GumbelSoftmax(tau=0.5, dim=1)
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, tau=1, dim=None, inplace=False):
        """Create a ``GumbelSoftmax`` module.

        Parameters
        ----------
        tau : Union[number, dragon.vm.torch.Tensor], default=1
            The temperature to use.
        dim : int, required
            The dimension to reduce.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.dim = dim
        self.inplace = inplace
        if dim is None:
            raise ValueError('Excepted a valid dim, got None.')

    def forward(self, logits=None, probs=None):
        if probs is not None:
            input = probs.log()
        else:
            input = logits - logits.logsumexp(dim=self.dim, keepdim=True)
        u_dist = init_funcs.rand(input.shape, dtype=input.dtype, device=input.device)
        gumbels = -((-(u_dist.log())).log())
        scores = (input + gumbels) / self.tau
        return F.softmax(scores, self.dim, self.inplace)


class LeakyReLU(Module):
    r"""Apply the leaky rectified linear unit.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                slope * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    m = torch.nn.LeakyReLU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.leaky_relu(...)`_ - Apply the leaky rectified linear unit.

    """

    def __init__(self, negative_slope=0.01, inplace=False):
        """Create a ``LeakyReLU`` module.

        Parameters
        ----------
        negative_slope : float, optional, default=0.01
            The slope of negative side.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)


class LogSoftmax(Module):
    r"""Apply the composite of logarithm and softmax.

    The **LogSoftmax** function is defined as:

    .. math:: \text{LogSoftmax}(x) = \log(\frac{\exp(x_{i})}{\sum \exp(x_{j})})

    Examples:

    ```python
    m = torch.nn.LogSoftmax(dim=1)
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, dim):
        """Create a ``LogSoftmax`` module.

        Parameters
        ----------
        dim : int
            The dimension to reduce.

        """
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

    def forward(self, input):
        return F.log_softmax(input, self.dim)


class PReLU(Module):
    r"""Apply the parametric rectified linear unit.
    `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    The **PReLU** function is defined as:

    .. math::
        \text{PReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                weight * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    # Use a single parameter to scale all channels
    # Typically known as the ``channel-shared`` style
    m = torch.nn.PReLU(num_parameters=1)
    x = torch.randn(2, 3)
    y = m(x)

    # Use different parameter for each channel
    mm =  torch.nn.PReLU(num_parameters=3)
    z = mm(x)
    ```
    """

    def __init__(self, num_parameters=1, init=0.25):
        """Create a ``PReLU`` module.

        Parameters
        ----------
        num_parameters : int, optional, default=1
            The number of parameters.
        init : float, optional, default=0.25
            The default value of parameters.

        """
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(Tensor(num_parameters).fill_(init))

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)

    def forward(self, input):
        return F.prelu(input, self.weight)


class ReLU(Module):
    r"""Apply rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \text{ReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    m = torch.nn.ReLU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, inplace=False):
        """Create a ``ReLU`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(ReLU, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)


class ReLU6(Module):
    r"""Apply the clipped-6 rectified linear unit.
    `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

    The **ReLU-6** function is defined as:

    .. math::
        \text{ReLU-6}(x) =
            \begin{cases}
                \min(x, 6), & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    m = torch.nn.ReLU6()
    x = torch.tensor([-2, 0, 2, 4, 6, 8], 'float32')
    y = m(x)
    ```

    """

    def __init__(self, inplace=False):
        """Create a ``ReLU6`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(ReLU6, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return F.relu6(input, inplace=self.inplace)


class SELU(Module):
    r"""Apply the scaled exponential linear unit.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    The **SELU** function is defined as:

    .. math::
        \text{SELU}(x) = 1.0507 *
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                1.67326 * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    m = torch.nn.SELU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, inplace=False):
        """Create a ``SELU`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(SELU, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return F.selu(input, self.inplace)


class Sigmoid(Module):
    r"""Apply the sigmoid function.

    The **Sigmoid** function is defined as:

    .. math:: \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    m = torch.nn.Sigmoid()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, inplace=False):
        """Create a ``Sigmoid`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return F.sigmoid(input, self.inplace)


class Softmax(Module):
    r"""Apply the softmax function.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

    Examples:

    ```python
    m = torch.nn.Softmax(dim=1)
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, dim, inplace=False):
        """Create a ``Softmax`` module.

        Parameters
        ----------
        dim : int
            The dimension to reduce.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Softmax, self).__init__()
        self.dim = dim
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, input):
        return F.softmax(input, self.dim, self.inplace)


class Tanh(Module):
    r"""Apply the tanh function.

    The **Tanh** function is defined as:

    .. math:: \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}

    Examples:

    ```python
    m = torch.nn.Tanh()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, inplace=False):
        """Create a ``Tanh`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Tanh, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return F.tanh(input, inplace=self.inplace)
