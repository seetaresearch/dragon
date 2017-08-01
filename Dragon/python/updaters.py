# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import pprint
from dragon.config import logger
import dragon.core.workspace as ws
from dragon.core.tensor import Tensor

class Updater(object):
    def __init__(self,
                 scale_gradient = 1.0,
                 clip_gradient = -1.0,
                 l2_decay = -1.0,
                 slot=''):

        self._hyper_params = {'scale_gradient': scale_gradient,
                              'clip_gradient': clip_gradient,
                              'l2_decay': l2_decay}
        self._extra_kwargs = {'slot': slot}
        self._tuples = []
        self._type = None
        self._prefix = ''

    def append(self, tuple, lr_mult = 1.0, decay_mult = 1.0):

        """
        :param tuple:           tuple[0] is value Tensor
                                tuple[1] is gradient Tensor
        :param lr_mult:         lr = base_lr * lr_mult
        :param decay_mult:      decay = weight_decay * decay_mult
        """

        tensors = (tensor.name if isinstance(tensor, Tensor) \
                        else tensor for tensor in tuple )
        kwargs = { 'lr_mult': lr_mult, 'decay_mult': decay_mult }
        self._tuples.append((tensors, kwargs))

    @property
    def lr(self):
        return ws.FetchTensor(self._prefix + 'base_lr')[0]

    @lr.setter
    def lr(self, lr):
        ws.FeedTensor(self._prefix + 'base_lr', np.array([lr], dtype=np.float32))

    def echo(self):
        logger.info('---------------------------------------------------------')
        logger.info('Optimizer: {}, Using config:'.format(self._type.split('Update')[0]))
        pprint.pprint(self._hyper_params)
        logger.info('---------------------------------------------------------')


class SGDUpdater(Updater):
    def __init__(self, base_lr=0.01, momentum=0.9, **kwargs):
        super(SGDUpdater, self).__init__(**kwargs)
        self._hyper_params = dict({'base_lr': base_lr,
                                   'momentum': momentum},
                                   **self._hyper_params)
        self._type = 'SGDUpdate'
        self.echo()


class NesterovUpdater(Updater):
    def __init__(self, base_lr=0.01, momentum=0.9, **kwargs):
        super(NesterovUpdater, self).__init__(**kwargs)
        self._hyper_params = dict({'base_lr': base_lr,
                                   'momentum': momentum},
                                  **self._hyper_params)
        self._type = 'NesterovUpdate'
        self.echo()


class RMSPropUpdater(Updater):
    def __init__(self, base_lr=0.01, decay=0.9, eps=1e-8, **kwargs):
        super(RMSPropUpdater, self).__init__(**kwargs)
        self._hyper_params = dict({'base_lr': base_lr,
                                   'decay': decay,
                                   'eps': eps},
                                   **self._hyper_params)
        self._type = 'RMSPropUpdate'
        self.echo()


class AdamUpdater(Updater):
    def __init__(self, base_lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super(AdamUpdater, self).__init__(**kwargs )
        self._hyper_params = dict({'base_lr': base_lr,
                                   'beta1': beta1, 'beta2': beta2,
                                   'eps': eps},
                                   **self._hyper_params)
        self._type = 'AdamUpdate'
        self.echo()