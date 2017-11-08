# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import pprint
import dragon.core.workspace as ws
from dragon.core.tensor import Tensor

class BaseUpdater(object):
    """
    BaseUpdater is designed to preprocess the gradients.
    """
    def __init__(self, scale_gradient = 1.0, clip_gradient = -1.0,
                 l2_decay = -1.0, slot=''):
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
        self._hyper_params = {'scale_gradient': scale_gradient,
                              'clip_gradient': clip_gradient,
                              'l2_decay': l2_decay}
        self._extra_kwargs = {'slot': slot}
        self._tuples = []
        self._type = None
        self._prefix = ''

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
        tensors = (tensor.name if isinstance(tensor, Tensor) \
                        else tensor for tensor in pair )
        arguments = {'lr_mult': lr_mult, 'decay_mult': decay_mult}
        self._tuples.append((tensors, arguments))

    @property
    def lr(self):
        """Set or get the learning rate.

        Parameters
        ----------
        learning_rate : basic numerical type
            The learning rate to set.

        Returns
        -------
        basic numerical type
            The learning rate that this updater has currently applied.

        """
        return ws.FetchTensor(self._prefix + 'base_lr')[0]

    @lr.setter
    def lr(self, lr):
        ws.FeedTensor(self._prefix + 'base_lr', np.array([lr], dtype=np.float32))

    def echo(self):
        """
        Print Updater Information.
        """
        from dragon.config import logger
        logger.info('---------------------------------------------------------')
        logger.info('Optimizer: {}, Using config:'.format(self._type.split('Update')[0]))
        pprint.pprint(self._hyper_params)
        logger.info('---------------------------------------------------------')


class SGDUpdater(BaseUpdater):
    """
    The Momentum-SGD Updater, introduced by `[LeCun et.al, 1998] <http://yann.lecun.com/exdb/publis/#lecun-98b>`_.
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
        self._hyper_params = dict({'base_lr': base_lr,
                                   'momentum': momentum},
                                   **self._hyper_params)
        self._type = 'SGDUpdate'
        self.echo()


class NesterovUpdater(BaseUpdater):
    """
    The Nesterov-SGD Updater, introduced by `[Sutskever et.al, 2012] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.
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
        self._hyper_params = dict({'base_lr': base_lr,
                                   'momentum': momentum},
                                  **self._hyper_params)
        self._type = 'NesterovUpdate'
        self.echo()


class RMSPropUpdater(BaseUpdater):
    """
    The RMSProp Updater, introduced by `[Hinton et.al, 2013] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.
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
        self._hyper_params = dict({'base_lr': base_lr,
                                   'decay': decay,
                                   'eps': eps},
                                   **self._hyper_params)
        self._type = 'RMSPropUpdate'
        self.echo()


class AdamUpdater(BaseUpdater):
    """
    The Adam Updater, introduced by `[Kingma & Ba, 2014] <https://arxiv.org/abs/1412.6980>`_.
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
        self._hyper_params = dict({'base_lr': base_lr,
                                   'beta1': beta1, 'beta2': beta2,
                                   'eps': eps},
                                   **self._hyper_params)
        self._type = 'AdamUpdate'
        self.echo()