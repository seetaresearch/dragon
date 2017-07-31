# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Train a FCN-32s(PASCAL VOC) network """

import dragon.vm.caffe as caffe
import surgery
import numpy as np

weights = '../transplants/VGG16/VGG16.fcn.caffemodel'

if __name__ == '__main__':

    # init
    caffe.set_mode_gpu()
    caffe.set_device(0)

    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from(weights)

    # surgeries
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

    for _ in range(25):
        solver.step(4000)