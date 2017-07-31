# Fully Convolutional Networks for Semantic Segmentation

This is the reference implementation of the models and code for the fully convolutional networks (FCNs) in the [PAMI FCN](https://arxiv.org/abs/1605.06211) and [CVPR FCN](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) papers:

    Fully Convolutional Models for Semantic Segmentation
    Evan Shelhamer*, Jonathan Long*, Trevor Darrell
    PAMI 2016
    arXiv:1605.06211

    Fully Convolutional Models for Semantic Segmentation
    Jonathan Long*, Evan Shelhamer*, Trevor Darrell
    CVPR 2015
    arXiv:1411.4038

### Requirements: software

Python packages you might not have: `numpy`, `PIL`, `python-opencv`

### Requirements: hardware

For training the FCN with VGG16 for VOC images(~500x350), 4G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. We'll call the directory of Seg-FCN as `FCN_ROOT`

2. Download pre-computed Seg-FCN models

* [FCN-32s PASCAL](http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel): single stream, 32 pixel prediction stride net, scoring 63.6 mIU on seg11valid
* [FCN-16s PASCAL](http://dl.caffe.berkeleyvision.org/fcn16s-heavy-pascal.caffemodel): two stream, 16 pixel prediction stride net, scoring 65.0 mIU on seg11valid
* [FCN-8s PASCAL](http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel): three stream, 8 pixel prediction stride net, scoring 65.5 mIU on seg11valid and 67.2 mIU on seg12test
* [FCN-8s PASCAL at-once](http://dl.caffe.berkeleyvision.org/fcn8s-atonce-pascal.caffemodel): all-at-once, three stream, 8 pixel prediction stride net, scoring 65.4 mIU on seg11valid

```Shell
    cp fcn8s-heavy-pascal.caffemodel $FCN_ROOT/data/seg_fcn_models
```

These models were trained online with high momentum, using extra data from [Hariharan et al.](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html), but excluding SBD val.

FCN-32s is fine-tuned from the [ILSVRC-trained VGG-16 model](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014), and the finer strides are then fine-tuned in turn.

The "at-once" FCN-8s is fine-tuned from VGG-16 all-at-once by scaling the skip connections to better condition optimization.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FCN_ROOT
python infer.py
```
The demo performs semantic segmentation using a VGG16 network trained for semantic segmentation on SBDD.

### Beyond the demo: installation for training and testing models
1. Download the SBDD(for training), VOC2011(for testing)

	```Shell
	wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
	```

2. Extract all of these tars into the $FCN_ROOT/data, should have this basic structure

	```Shell
  	$FCN_ROOT/data/sbdd/dataset
  	$FCN_ROOT/data/pascal/VOC2011
  	# ... and several other directories ...
  	```
3. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained [ImageNet models](http://pan.baidu.com/s/1eSGLwsE) can be downloaded for backbone net: VGG16.

### Transplant a fully-connected net into a fully-convolution net

```Shell
cp VGG16.v2.caffemodel $FCN_ROOT/transplant/VGG16
cd $FCN_ROOT/transplant/VGG16
python solve.py
```
This script will generate a new model ``VGG16.fcn.caffemodel`` for training.


### Training

FCN prefers two training methods:

1. CVPR version:

    First, Train FCN-32s for 1 day.

    Then, Train FCN-16s fintune from FCN-32s for 1 day.

    Final, Train FCN-8s fintune from FCN-16s for 1 day.

    Follow this way, you should run $FCN_ROOT/voc-fcn32s | voc-fcn16s | fcn-8s/solve.py ``sequentially``.

2. PAMI version:

    Directly run $FCN_ROOT/voc-fcn8s-atonce/solve.py

Both of above ways train same iterations, ``PAMI ver.`` is simpier and got similar results.


Trained Seg-FCN networks are saved under:

```
voc-fcnxs/snapshot/
```

Test outputs are saved under:

```
voc-fcnxs/segs/
```
