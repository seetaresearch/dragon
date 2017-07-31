# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Transplant fully-connected caffemodel into fully-convolution ver. """

import surgery
import dragon.vm.caffe as caffe

if __name__ == '__main__':

    net = caffe.Net('net.prototxt', 'VGG16.v2.caffemodel', caffe.TEST)
    new_net = caffe.Net('new_net.prototxt', caffe.TEST)
    surgery.transplant(new_net, net)
    new_net.save('VGG16.fcn.caffemodel', suffix='')