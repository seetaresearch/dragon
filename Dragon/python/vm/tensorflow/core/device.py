# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = ['device']

class device(object):
    def __init__(self, dev):
        assert isinstance(dev, str)
        self.device = dev.split('/')[-1].split(':')[0]
        self.id = dev.split('/')[-1].split(':')[1]

    def __enter__(self):
        import dragon.core.scope as scope
        scope.DEVICE_SCOPE = '/' + self.device + ':' + str(self.id)

    def __exit__(self, type, value, traceback):
        import dragon.core.scope as scope
        scope.DEVICE_SCOPE = ''