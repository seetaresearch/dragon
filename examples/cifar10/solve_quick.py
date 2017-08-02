# --------------------------------------------------------
# Cifar-10 for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Train a cifar-10 net """

import dragon.vm.caffe as caffe

if __name__ == '__main__':

    # init
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # solve
    solver = caffe.SGDSolver('cifar10_quick_solver.prototxt')
    solver.step(5000)
    solver.snapshot()
