# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from six.moves import range as xrange

from dragon.vm.tensorflow.core.variables import placeholder


def feed_check(feed_dict):
    if feed_dict is not None:
        for key, value in feed_dict.items():
            if type(key) != placeholder:
                raise TypeError('only a placeholder can be feeded.')
            if key.shape is not None:
                if len(key.shape) != len(value.shape):
                    raise RuntimeError('placeholder limits {} dims, but feed {}'
                                       .format(len(key.shape), len(value.shape)))
                for i in xrange(len(key.shape)):
                    if key.shape[i] is None: continue
                    if key.shape[i] != value.shape[i]:
                        raise RuntimeError('placeholder limits shape as (' +
                                           ','.join([str(dim) for dim in key.shape]) + '), ' +
                                           'but feed (' + ','.join([str(dim) for dim in value.shape]) + ').')