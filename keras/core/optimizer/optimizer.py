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
"""Basic optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.training import optimizer as optimizer_v1


class Optimizer(optimizer_v1.Optimizer):
    """The base class for optimizers."""

    def __init__(self, name=None, **kwargs):
        """Create an ``Optimizer``.

        Parameters
        ----------
        name : str, optional
            The optional optimizer name.

        """
        self._name = name
        clip_norm = kwargs.pop('clipnorm', 0)
        clip_value = kwargs.pop('clipvalue', 0)
        clip_norm = kwargs.pop('global_clipnorm', clip_norm)
        super(Optimizer, self).__init__(clip_norm=clip_norm,
                                        clip_value=clip_value)
        self._iterations = 0
        self._hyper_aliases = {'clipnorm': 'clip_norm',
                               'clipvalue': 'clip_value',
                               'global_clipnorm': 'clip_norm'}

    @property
    def iterations(self):
        """Return the number of steps has run.

        Returns
        -------
        int
            The iterations.

        """
        return self._iterations

    @iterations.setter
    def iterations(self, variable):
        self._iterations = variable

    def apply_gradients(self, grads_and_vars):
        """Apply the gradients to update variables.

        Parameters
        ----------
        grads_and_vars : Sequence[Sequence[dragon.Tensor]]
            The gradients and variables.

        Returns
        -------
        dragon.vm.tensorflow.keras.optimizers.Optimizer
            The self to generate the update operations.

        """
        super(Optimizer, self).apply_gradients(grads_and_vars)
        self._iterations += 1
        return self

    def __getattr__(self, item):
        aliases = self.__dict__.get('_hyper_aliases')
        item = aliases.get(item, item)
        return super(Optimizer, self).__getattr__(item)

    def __setattr__(self, key, value):
        aliases = self.__dict__.get('_hyper_aliases')
        if aliases:
            key = aliases.get(key, key)
        super(Optimizer, self).__setattr__(key, value)
