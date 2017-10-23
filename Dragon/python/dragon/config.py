# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.__init__ import *

import logging
logger = logging.getLogger('dragon')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

option = {}

REGISTERED_OPERATORS = set(s for s in RegisteredOperatorsCC())
NO_GRADIENT_OPERATORS = set(s for s in NoGradientOperatorsCC())

# The current device, 'CPU' or 'CUDA'
option['device'] = 'CPU'

# The device id
option['gpu_id'] = 0

# Whether to use cuDNN if possible
option['use_cudnn'] = False

# The global random seed
option['random_seed'] = 3

# Disable the memonger if true
option['debug_mode'] = False

# Set it by the memonger
option['share_grads'] = False


def EnableCPU():
    """Enable CPU mode globally.

    Returns
    -------
    None

    """
    global option
    option['device'] = 'CPU'


def EnableCUDA(gpu_id=0, use_cudnn=True):
    """Enable CUDA mode globally.

    Parameters
    ----------
    gpu_id : int
        The id of GPU to use.
    use_cudnn : boolean
        Whether to use cuDNN if available.

    Returns
    -------
    None

    """
    global option
    option['device'] = 'CUDA'
    option['gpu_id'] = gpu_id
    option['use_cudnn'] = use_cudnn

# TODO(PhyscalX): please not use @setter
# TODO(PhyscalX): seems that it can't change the global value


def SetRandomSeed(seed):
    """Set the global random seed.

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    None

    """
    global option
    option['random_seed'] = seed


def GetRandomSeed():
    """Get the global random seed.

    Returns
    -------
    int
        The global random seed.

    """
    global option
    return option['random_seed']


def SetGPU(id):
    """Set the global id GPU.

    Parameters
    ----------
    id : int
        The id of GPU to use.

    Returns
    -------
    None

    """
    global option
    option['gpu_id'] = id


def GetGPU(id):
    """Get the global id of GPU.

    Returns
    -------
    int
        The global id of GPU.

    """
    global option
    return option['gpu_id']


def SetDebugMode(enabled=True):
    """Enable Debug mode globally.

    It will disable all memory sharing optimizations.

    Parameters
    ----------
    enabled : boolean
        Whether to enable debug mode.

    Returns
    -------
    None

    """
    global option
    option['debug_mode'] = enabled


def SetLoggingLevel(level):
    """Set the minimum level of Logging.

    Parameters
    ----------
    level : str
        The level, ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``FATAL``.

    Notes
    -----
    The default level is ``INFO``.

    """
    SetLogLevelCC(level)
    global logger
    logger.setLevel({
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'FATAL': logging.CRITICAL
    }[level])