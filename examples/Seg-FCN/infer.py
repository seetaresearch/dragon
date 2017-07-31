# --------------------------------------------------------
# Seg-FCN for Dragon
# Copyright (c) 2017 SeetaTech
# Source Code by Evan Shelhamer
# Re-Written by Ting Pan
# --------------------------------------------------------

""" Infer for a single Image and show """

import numpy as np
from PIL import Image
import dragon.vm.caffe as caffe
import dragon.core.workspace as ws
import os
import cv2

# init
caffe.set_mode_gpu()
# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'data/seg_fcn_models/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# load color table
color_table = np.fromfile('colors/pascal_voc.act', dtype=np.uint8)

def load_image(file):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(file)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    return in_

def seg(file, save_dir="data/seg_results", mix=True, show=True):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    im = load_image(file)
    # shape for input (data blob is N x C x H x W), set data
    im = im.reshape(1, *im.shape)
    ws.FeedTensor(net.blobs['data'].data, im)

    # run net and take argmax for prediction
    net.forward()

    if save_dir is not None:
        filename_ext = file.split('/')[-1]
        filename = filename_ext.split('.')[-2]
        filepath = os.path.join(save_dir, filename + '.png')

        mat = ws.FetchTensor(net.blobs['score'].data)
        im = Image.fromarray(mat[0].argmax(0).astype(np.uint8), mode='P')
        im.putpalette(color_table)
        im.save(filepath)

        if show:
            if mix:
                show1 = cv2.imread(file)
                show2 = cv2.imread(filepath)
                show3 = cv2.addWeighted(show1, 0.7, show2, 0.5, 1)
            else: show3 = cv2.imread(filepath)
            cv2.imshow('Seg-FCN', show3)
            cv2.waitKey(0)

if __name__ == '__main__':

    seg('data/demo/001763.jpg')
