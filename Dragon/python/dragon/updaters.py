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

"""Define the update ops generator.

We dubbed them as ``Updater``, because ``Optimizer``
is used by so many Deep Learning frameworks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

from dragon.core import workspace
from dragon.core.tensor import Tensor


class BaseUpdater(object):
    """BaseUpdater is designed to pre-process the gradients."""

    # Store the global unique slot index
    _DEFAULT_UNIQUE_SLOT_ID = 0

    def __init__(self,
                 scale_gradient=1.0,
                 clip_gradient=-1.0,
                 l2_decay=-1.0,
                 slot=None,
                 verbose=True):
        """Construct a Updater to optimize the objectives.

        Parameters
        ----------
        scale_gradient : float
            The scale factor of gradients.
        clip_gradient : float
            The clip factor of gradients.
        l2_decay : float
            The l2 decay factor. Default is ``-1.0`` (Disabled).
        slot : str
            The slot name of advanced updater.

        """
        self._defaults = {
            'scale_gradient': scale_gradient,
            'clip_gradient': clip_gradient,
            'l2_decay': l2_decay,
        }
        self._param_group = []
        if slot: self._slot = slot
        else:
            BaseUpdater._DEFAULT_UNIQUE_SLOT_ID += 1
            self._slot = 'Updater/Slot:{}'.format(
                BaseUpdater._DEFAULT_UNIQUE_SLOT_ID)
        self._verbose = verbose
        self._registered = False
        self._extra_kwargs = {}

    def append(self, pair, lr_mult=1.0, decay_mult=1.0):
        """Append an ``UpdatePair`` into the updater.

        Parameters
        ----------
        pair : tuple or list
            The pair represent (values, grads).
        lr_mult : float
            The learning rate multiplier.
        decay_mult : float
            The decay factor multiplier.

        Returns
        -------
        None

        """
        pair = (tensor.name if isinstance(tensor, Tensor) \
            else tensor for tensor in pair)
        self._param_group.append((pair,
            {'lr_mult': lr_mult, 'decay_mult': decay_mult}))

    def __getattr__(self, item):
        defaults = self.__dict__.get('_defaults')
        if item in defaults:
            if self._registered:
                return workspace.FetchTensor(self._slot + '/' + item)
            else: return defaults[item]
        return self.__dict__[item]

    def __setattr__(self, key, value):
        defaults = self.__dict__.get('_defaults')
        if defaults is not None and key in defaults:
            if self._registered:
                workspace.FeedTensor(self._slot + '/' + key, value,
                    dtype='float32', force_cpu=True)
            else:
                self._defaults[key] = value
        else:
            object.__setattr__(self, key, value)

    def register_in_workspace(self):
        if not self._registered:
            for k, v in self._defaults.items():
                workspace.FeedTensor(self._slot + "/" + k, v,
                    dtype='float32', force_cpu=True)
            self._registered = True
            if self._verbose:
                print('---------------------------------------------------------')
                print('Optimizer: {}, Using config:'.format(self.type(True)))
                pprint.pprint(self._defaults)
                print('---------------------------------------------------------')

    def type(self, no_suffix=False):
        return self.__class__.__name__.split('Updater')[0] \
            if no_suffix else self.__class__.__name__[0:-1]


class SGDUpdater(BaseUpdater):
    """The Momentum-SGD Updater.

    Introduced by `[LeCun et.al, 1998] <http://yann.lecun.com/exdb/publis/#lecun-98b>`_.

    """
    def __init__(self, base_lr=0.01, momentum=0.9, **kwargs):
        """Construct a Momentum-SGD Updater to optimize the objectives.

        Parameters
        ----------
        base_lr : float
            The base learning rate.
        momentum : float
            The momentum.

        """
        super(SGDUpdater, self).__init__(**kwargs)
        self._defaults = dict({
            'base_lr': base_lr,
            'momentum': momentum
        }, **self._defaults)


class NesterovUpdater(BaseUpdater):
    """The Nesterov-SGD Updater.

    Introduced by `[Sutskever et.al, 2012] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.

    """
    def __init__(self, base_lr=0.01, momentum=0.9, **kwargs):
        """Construct a Nesterov-SGD Updater to optimize the objectives.

        Parameters
        ----------
        base_lr : float
            The base learning rate.
        momentum : float
            The momentum.

        """
        super(NesterovUpdater, self).__init__(**kwargs)
        self._defaults = dict({
            'base_lr': base_lr,
            'momentum': momentum
        }, **self._defaults)


class RMSPropUpdater(BaseUpdater):
    """The RMSProp Updater.

    Introduced by `[Hinton et.al, 2013] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.

    """
    def __init__(self, base_lr=0.01, decay=0.9, eps=1e-8, **kwargs):
        """Construct a RMSProp Updater to optimize the objectives.

        Parameters
        ----------
        base_lr : float
            The base learning rate.
        decay : float
            The decay.
        eps : float
            The eps.

        """
        super(RMSPropUpdater, self).__init__(**kwargs)
        self._defaults = dict({
            'base_lr': base_lr,
            'decay': decay,
            'eps': eps
        }, **self._defaults)


class AdamUpdater(BaseUpdater):
    """The Adam Updater.

    Introduced by `[Kingma & Ba, 2014] <https://arxiv.org/abs/1412.6980>`_.

    """
    def __init__(self, base_lr=0.01, beta1=0.9,
                 beta2=0.999, eps=1e-8, **kwargs):
        """Construct a Adam Updater to optimize the objectives.

        Parameters
        ----------
        base_lr : float
            The base learning rate.
        beta1 : float
            The beta1.
        beta2 : float
            The beta2.
        eps : float
            The eps.

        """
        super(AdamUpdater, self).__init__(**kwargs )
        self._defaults = dict({
            'base_lr': base_lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps
        }, **self._defaults)