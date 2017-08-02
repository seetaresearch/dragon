# --------------------------------------------------------
# Cifar-10 for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

""" Infer for a single Image and show """

import dragon.vm.caffe as caffe
import numpy as np
import cv2

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# init
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
net = caffe.Net("cifar10_quick_deploy.prototxt",
                'snapshots/cifar10_quick_iter_5000.caffemodel', caffe.TEST)


def load_image(filename):
    # load image, subtract mean, and make dims 1 x 1 x H x W
    im = cv2.imread(filename)
    im = cv2.resize(im, (32, 32))
    im = np.array(im, dtype=np.float32)
    im -= np.array((104.0, 116.0, 122.0))
    im = im.transpose((2,0,1))
    return im[np.newaxis, :, :, :]


def run(filename):

    # infer
    im = load_image(filename)
    net.forward(**{'data': im})
    score = net.blobs['ip2'].data.get_value()[0]
    pred = score.argmax(0)

    # show
    print(classes[pred])


if __name__ == '__main__':

    run('data/demo/cat.jpg')
