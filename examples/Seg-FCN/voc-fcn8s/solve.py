# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Train a FCN-8s(PASCAL VOC) network """

import dragon.vm.caffe as caffe
import surgery

weights = '../voc-fcn16s/snapshot/train_iter_100000.caffemodel'

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