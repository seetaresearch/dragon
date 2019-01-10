# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Please convert models at python2.7,

as the protocols of python3.x of pickle are not downward compatible.

"""

import os
import sys
from collections import OrderedDict

import torch
from torch.utils.model_zoo import \
    _download_url_to_file, urlparse, HASH_REGEX


def state_dict_v2(m, destination=None, prefix=''):
    """A forked version of nn.Module.state_dict().

    We apply a tensor2numpy for each parameters recursively,
    which is compatible with dragon.vm.torch.nn.Module.load_state_dict().

    Parameters
    ----------
    m : dragon.vm.torch.nn.Module
        The fb.pytorch module.
    destination : OrderedDict, optional
        The output dict.
    prefix : str, optional
        The prefix to the key of dict.

    Returns
    -------
    OrderedDict
        The dict with numpy parameters.

    """
    t2np = lambda t : t.cpu().numpy()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    for name, param in m._parameters.items():
        if param is not None:
            destination[prefix + name] = t2np(param.data)
    for name, buf in m._buffers.items():
        if buf is not None:
            destination[prefix + name] = t2np(buf)
    for name, module in m._modules.items():
        if module is not None:
            state_dict_v2(module, destination, prefix + name + '.')
    return destination


if __name__ == '__main__':
    from torchvision.models import (
        alexnet,
        vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
        resnet18, resnet34, resnet50, resnet101, resnet152,
        squeezenet1_0, squeezenet1_1,
        inception_v3,
    )

    tasks = OrderedDict([
        # (alexnet, 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'),
        (vgg11, 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'),
        (vgg13, 'https://download.pytorch.org/models/vgg13-c768596a.pth'),
        # (vgg16, 'https://download.pytorch.org/models/vgg16-397923af.pth'),
        # (vgg16, 'https://download.pytorch.org/models/vgg16-397923af.pth'),
        # (vgg16, 'https://download.pytorch.org/models/vgg16-397923af.pth'),
        # (vgg19, 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'),
        (vgg11_bn, 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'),
        (vgg13_bn, 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'),
        (vgg16_bn, 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'),
        (vgg19_bn, 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'),
        # (resnet18, 'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
        # (resnet34, 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
        # (resnet50, 'https://download.pytorch.org/models/resnet50-19c8e357.pth'),
        # (resnet101, 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
        # (resnet152, 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
        # (squeezenet1_0, 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth'),
        # (squeezenet1_1, 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'),
        # (inception_v3, 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'),
    ])

    downloads = []
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    for m, url in tasks.items():
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            hash_prefix = HASH_REGEX.search(filename).group(1)
            _download_url_to_file(url, cached_file, hash_prefix, progress=True)
        downloads.append((m, cached_file))

    if sys.version_info >= (3,0):
        raise RuntimeError('You can download with python2|3, but convert with python2 only.')

    import cPickle
    for m, file in downloads:
        p1, p2 = os.path.split(file)
        p3, p4 = os.path.splitext(p2)
        file_v2 = os.path.join(p1, p3 + '.pkl')
        mi = m()
        mi.load_state_dict(torch.load(file))
        np_state_dict = state_dict_v2(mi)
        cPickle.dump(np_state_dict, open(file_v2, 'wb'), cPickle.HIGHEST_PROTOCOL)
        print('Convert {} to {}.'.format(file, file_v2))