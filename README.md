# Dragon: A Computation Graph Virtual Machine Based Deep Learning Framework
![](http://dragon.seetatech.com/static/images/styles-dragon.png)
-----
([Installation](http://dragon.seetatech.com/helper/install.html) | [Documentation](http://dragon.seetatech.com/api/python/index.html))
-----
Dragon is a **C**(Computation)**G**(Graph)**V**(Virtual)**M**(Machine) based distributed deep learning framework.

Our goal is to reduce the unnecessary structures or interfaces. Therefore, in addition to feed or fetch, the last thing is designing a objective function through all available operators.

Besides, we demonstrate that a cross-frameworks frontend(**Deep Learning VirtualBox**) is feasible, and further more, will get benefit from all participating crucial interfaces especially when one is not reasonable.

## WHY NOT Original DL Frameworks?

I was always confused in my childhood of studying DeepLearning:
 
```python
import theano
import caffe
import tensorflow
import torch
```
Too stupied, ISN'T?

One day, I saw a JOKE:

```python
# FXCK TF
# KEEP CALM AND USE PYTORCH
import tensorflow as torch
```

So, I made it:

```python
import dragon.vm.theano as theano
import dragon.vm.caffe as caffe
import dragon.vm.tensorflow as tensorflow
import dragon.vm.torch as torch
```

WOW, I could use ALL above DL Frameworks all together!

## News

Dragon 0.2.2 Released -  Cleaner, Faster, Stronger and now we have DYNAMIC GRAPH >>> (VM.PyTorch :-) <<<

<div align=center><img width="373" height="214" src="http://dragon.seetatech.com/static/images/dragon-0.2.2.png"/></div>

## License and Citation
Dragon is released under the [BSD 2-Clause license](https://github.com/neopenx/Dragon/blob/master/LICENSE).

Please cite Dragon in your publications if it helps your research:

    @article{pan2017dragon,
      Author = {Pan, Ting},
      Journal = {arXiv preprint arXiv:1707.08265},
      Title = {Dragon: A Computation Graph Virtual Machine Based Deep Learning Framework},
      Year = {2017}
    }
