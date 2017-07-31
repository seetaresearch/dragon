# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Test a FCN-32s(PASCAL VOC) network """

import dragon.vm.caffe as caffe
import score
import numpy as np

weights = 'snapshot/train_iter_100000.caffemodel'

if __name__ == '__main__':

    # init
    caffe.set_mode_gpu()
    caffe.set_device(0)

    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from(weights)

    # scoring
    val = np.loadtxt('../data/seg11valid.txt', dtype=str)
    score.seg_tests(solver, 'seg', val)

