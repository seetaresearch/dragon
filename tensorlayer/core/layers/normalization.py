# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
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

from dragon.core.ops import normalization_ops
from dragon.vm.tensorlayer.core.engine import layer
from dragon.vm.tensorlayer.core.layers import utils


class BatchNorm(layer.Layer):
    r"""The layer to apply the batch normalization.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    Examples:

    ```python
    x = tl.layers.Input([None, 32, 50, 50])
    y = tl.layers.BatchNorm()(x)
    ```

    """

    def __init__(
        self,
        decay=0.9,
        epsilon=1e-5,
        act=None,
        beta_init='zeros',
        gamma_init='ones',
        moving_mean_init='zeros',
        moving_var_init='ones',
        num_features=None,
        data_format='channels_first',
        name=None,
    ):
        """Create a ``BatchNorm`` layer.

        Parameters
        ----------
        decay : float, optional, default=0.9
            The decay factor for moving average.
        epsilon : float, optional, default=1e-5
            The epsilon.
        act : callable, optional
            The optional activation function.
        beta_init : Union[callable, str], optional
            The initializer for ``beta``.
        gamma_init : Union[callable, str], optional
            The initializer for ``gamma``.
        moving_mean_init : Union[callable, str], optional
            The initializer for ``moving_mean``.
        moving_var_init : Union[callable, str], optional
            The initializer for ``moving_var``.
        num_features: int, optional
            The number of input features.
        data_format : {'channels_first', 'channels_last'}, optional
             The optional data format.
        name : str, optional
            The optional layer name.

        """
        super(BatchNorm, self).__init__(name, act)
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.num_features = num_features

        self.beta = None
        self.gamma = None
        self.moving_mean = None
        self.moving_var = None

        if self.data_format == 'channels_last':
            self.axis = -1
        elif self.data_format == 'channels_first':
            self.axis = 1
        else:
            raise ValueError(
                'data_format should be either %s or %s' %
                ('channels_last', 'channels_first')
            )

        if num_features is not None:
            self.build(None)

        if self.decay < 0. or 1. < self.decay:
            raise ValueError("decay should be between 0 to 1")

    def __repr__(self):
        """
        Return a repr representation of this field.

        Args:
            self: (todo): write your description
        """
        s = '{classname}(' \
            'num_features={num_features}, ' \
            'decay={decay}, ' \
            'epsilon={epsilon}'
        s += (', ' + utils.get_act_str(self.act))
        if self.name is not None:
            s += ', name="{name}"'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, input_shape):
        """
        Connects the graph.

        Args:
            self: (todo): write your description
            input_shape: (list): write your description
        """
        if input_shape is None:
            num_features = self.num_features
        else:
            num_features = input_shape[self.axis]
        self.beta = \
            self.add_weight(
                name='beta',
                shape=[num_features],
                init=self.beta_init if self.beta_init else 'zeros',
                trainable=True if self.beta_init else False,
            )
        self.gamma = \
            self.add_weight(
                name="gamma",
                shape=[num_features],
                init=self.gamma_init if self.gamma_init else 'ones',
                trainable=True if self.gamma_init else False,
            )
        self.moving_mean = \
            self.add_weight(
                name='moving_mean',
                shape=[num_features],
                init=self.moving_mean_init,
                trainable=False,
            )
        self.moving_var = \
            self.add_weight(
                name="moving_var",
                shape=[num_features],
                init=self.moving_var_init,
                trainable=False,
            )
        self._built = True

    def forward(self, inputs, **kwargs):
        """
        Perform forward

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        outputs = normalization_ops.batch_norm(
            [inputs,
             self.gamma,
             self.beta,
             self.moving_mean,
             self.moving_var],
            axis=self.axis,
            momentum=self.decay,
            epsilon=self.epsilon,
            use_stats=0 if self.training else 1,
        )
        if self.act:
            outputs = self.act(outputs)
        return outputs
