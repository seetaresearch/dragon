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
"""Activation modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.init import xavier_uniform_
from dragon.vm.torch.core.nn.modules.linear import Linear
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops import random_ops
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
    `torch.nn.functional.elu(...)`_

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


class GELU(Module):
    r"""Apply the gaussian error linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **GELU** function is defined as:

    .. math:: \text{GELU}(x) = x\cdot\frac{1}{2}[1 + \text{erf}(x / \sqrt{2})]

    Examples:

    ```python
    m = torch.nn.GELU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.gelu(...)`_

    """

    def __init__(self, approximate='none'):
        """Create a ``GELU`` module.

        Parameters
        ----------
        approximate : str, optional, default='none'
            The approximate algorithm.

        """
        super(GELU, self).__init__()
        self.approximate = approximate

    def extra_repr(self):
        return 'approximate={}'.format(self.approximate)

    def forward(self, input):
        return F.gelu(input, approximate=self.approximate)


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

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, input):
        u_dist = random_ops.rand(input.shape, dtype=input.dtype,
                                 device=input.device)
        gumbel = -((-(u_dist.log())).log())
        gumbel = (input + gumbel) / self.tau
        return F.softmax(gumbel, self.dim, self.inplace)


class Hardsigmoid(Module):
    r"""Apply the hard sigmoid function.

    The **HardSigmoid** function is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Examples:

    ```python
    m = torch.nn.Hardsigmoid()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.hardsigmoid(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``Hardsigmoid`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Hardsigmoid, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return F.hardsigmoid(input, self.inplace)


class Hardswish(Module):
    r"""Apply the hard swish function.
    `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

    The **HardSwish** function is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Examples:

    ```python
    m = torch.nn.Hardswish()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.hardswish(...)`_

    """

    def __init__(self):
        """Create a ``Hardswish`` module."""
        super(Hardswish, self).__init__()

    def forward(self, input):
        return F.hardswish(input)


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
    `torch.nn.functional.leaky_relu(...)`_

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

    See Also
    --------
    `torch.nn.functional.log_softmax(...)`_

    """

    def __init__(self, dim, inplace=False):
        """Create a ``LogSoftmax`` module.

        Parameters
        ----------
        dim : int
            The dimension to reduce.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(LogSoftmax, self).__init__()
        self.dim = dim
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, input):
        return F.log_softmax(input, self.dim, self.inplace)


class MultiheadAttention(Module):
    """Apply the multihead attention.
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    See Also
    --------
    `torch.nn.functional.multi_head_attention_forward(...)`_

    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.,
        bias=True,
        kdim=None,
        vdim=None,
    ):
        """Create a ``MultiheadAttention`` module.

        Parameters
        ----------
        embed_dim : int
            The dimension of input embeddings.
        num_heads : int
            The number of parallel heads.
        dropout: float, optional, default=0.
            The probability to set the attention to zero.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kdim : int, optional
            The dimension of key embedding.
        vdim : int, optional
            The dimension of value embedding.

        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError('<embed_dim> must be divisible by <num_heads>.')
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(Tensor(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            self.in_proj_bias.zero_()
            self.out_proj.bias.zero_()

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        return F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            dropout_p=self.dropout,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
        )


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

    See Also
    --------
    `torch.nn.functional.prelu(...)`_

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

    See Also
    --------
    `torch.nn.functional.relu(...)`_

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

    See Also
    --------
    `torch.nn.functional.relu6(...)`_

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

    See Also
    --------
    `torch.nn.functional.selu(...)`_

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

    See Also
    --------
    `torch.nn.functional.sigmoid(...)`_

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


class SiLU(Module):
    r"""Apply the sigmoid linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **SiLU** function is defined as:

    .. math:: \text{SiLU}(x) = x \cdot \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    m = torch.nn.So()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.silu(...)`_

    """

    def __init__(self):
        """Create a ``SiLU`` module."""
        super(SiLU, self).__init__()

    def forward(self, input):
        return F.silu(input)


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

    See Also
    --------
    `torch.nn.functional.softmax(...)`_

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

    See Also
    --------
    `torch.nn.functional.tanh(...)`_

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
