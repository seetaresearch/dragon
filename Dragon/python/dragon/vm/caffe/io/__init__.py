# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

def GetProperty(kwargs, name, default):
    return kwargs[name] \
        if name in kwargs else default