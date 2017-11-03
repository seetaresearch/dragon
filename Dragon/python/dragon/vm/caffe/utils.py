# --------------------------------------------------------
# Caffe @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

def ToFillerArgs(FillerParamerer):
    kwargs = \
    {'value' : FillerParamerer.value,
     'low': FillerParamerer.min,
     'high': FillerParamerer.max,
     'mean': FillerParamerer.mean,
     'std': FillerParamerer.std}
    return kwargs

