# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Source Code by Evan Shelhamer
# Re-Written by Ting Pan
# --------------------------------------------------------

from __future__ import division
import dragon.core.workspace as ws
import numpy as np

def transplant(new_net, net):
    # create graph
    net.function()
    new_net.function()

    for p in net.params:
        if p not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p]) - 1):
                print 'dropping', p, i
                break
            print 'copying', p, i
            net_param = ws.FetchTensor(net.params[p][i].data)
            new_net_param = ws.FetchTensor(new_net.params[p][i].data)
            name = new_net.params[p][i].data._name
            if net_param.shape != new_net_param.shape:
                print 'coercing', p, i, 'from', net_param.shape, 'to', new_net_param.shape
            else:
                pass
            new_net_param.flat = new_net_param.flat
            ws.FeedTensor(name, new_net_param)


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def interp(net, layers):
    print 'bilinear-interp for layers:', layers
    net.forward() # dragon must forward once to create weights
    for l in layers:
        net_param = ws.FetchTensor(net.params[l][0].data)
        m, k, h, w = net_param.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net_param[range(m), range(k), :, :] = filt
        ws.FeedTensor(net.params[l][0].data._name, net_param)


