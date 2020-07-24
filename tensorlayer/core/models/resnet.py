# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""ResNet for ImageNet.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
from dragon.vm import tensorlayer as tl


class Bottleneck(tl.models.Model):
    """The bottleneck block of resnet."""

    expansion = 4

    def __init__(self, dim_in, dim_out, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = tl.layers.Conv2d(
            in_channels=dim_in,
            n_filter=dim_out,
            filter_size=1,
            W_init='glorot_normal',
        )
        self.bn1 = tl.layers.BatchNorm(num_features=dim_out)
        self.conv2 = tl.layers.Conv2d(
            in_channels=dim_out,
            n_filter=dim_out,
            filter_size=3,
            strides=stride,
            W_init='glorot_normal',
        )
        self.bn2 = tl.layers.BatchNorm(num_features=dim_out)
        self.conv3 = tl.layers.Conv2d(
            in_channels=dim_out,
            n_filter=dim_out * self.expansion,
            filter_size=1,
            W_init='glorot_normal',
        )
        self.bn3 = tl.layers.BatchNorm(num_features=dim_out * self.expansion)
        self.downsample = downsample
        self.relu = tl.layers.Relu(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(tl.models.Model):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = tl.layers.Conv2d(
            in_channels=3,
            n_filter=self.inplanes,
            filter_size=7,
            strides=2,
            padding=3,
        )
        self.bn1 = tl.layers.BatchNorm(num_features=self.inplanes)
        self.relu = tl.layers.Relu(inplace=True)
        self.maxpool = tl.layers.MaxPool2d(filter_size=3, strides=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = tl.layers.GlobalMeanPool2d()
        self.fc = tl.layers.Dense(
            in_channels=512 * block.expansion,
            n_units=num_classes,
            b_init='zeros',
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = tl.layers.Flatten()(x)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tl.layers.LayerList([
                tl.layers.Conv2d(
                    in_channels=self.inplanes,
                    n_filter=planes * block.expansion,
                    filter_size=1,
                    strides=stride,
                ),
                tl.layers.BatchNorm(
                    num_features=planes * block.expansion
                ),
            ])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return tl.layers.LayerList(layers)


if __name__ == '__main__':
    dragon.autograph.set_execution('EAGER_MODE')
    m = ResNet(Bottleneck, [3, 4, 6, 3])
    x = tl.layers.Input((1, 3, 224, 224))
    y = m(x)
    print(y.shape)
