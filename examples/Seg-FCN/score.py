# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Source Code by Evan Shelhamer
# Re-Written by Ting Pan
# --------------------------------------------------------

from __future__ import division
import dragon.core.workspace as ws
import numpy as np
import os
from datetime import datetime
from PIL import Image

color_table = np.fromfile('../colors/pascal_voc.act', dtype=np.uint8)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = hist = None
    loss = 0
    for idx in dataset:
        net.forward()
        gt_mat = ws.FetchTensor(net.blobs[gt].data)
        layer_mat = ws.FetchTensor(net.blobs[layer].data)
        loss_mat = ws.FetchTensor(net.blobs['loss'].data)
        if n_cl is None: n_cl = layer_mat.shape[1]
        if hist is None: hist = np.zeros((n_cl, n_cl))
        hist += fast_hist(gt_mat[0, 0].flatten(),
                                layer_mat[0].argmax(0).flatten(), n_cl)

        if save_dir:
            im = Image.fromarray(layer_mat[0].argmax(0).astype(np.uint8), mode='P')
            im.putpalette(color_table)
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += loss_mat.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)


def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    if save_format:
        save_format = save_format.format(iter)
        if not os.path.exists(save_format): os.makedirs(save_format)

    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
