# Dragon: A Computation Graph Virtual Machine Based Deep Learning Framework

### Compile Requirements for C++

0. Google Protocol Buffer
1. Python (2 or 3, 64bit) &nbsp; | &nbsp; Anaconda (2 or 3, 64bit)
2. CUDA [Optional]
3. CUDNN [Optional]
4. OpenMPI [Optional]

-----
### Installation
1. Clone this repository

2. (Optional) Download and install [CUDA](https://developer.nvidia.com/cuda-toolkit)

      (Optional) Download and install [CUDNN](https://developer.nvidia.com/cudnn)

3. (Optional) Download 3rdparty.zip and unzip to Dragon/3rdparty (Out of source code dir)

    [*Win64-VS2013*](https://pan.baidu.com/s/1miGAZl2) (OpenBLAS / Protobuf2.6 for VS2013 / CUDNN v7 / Microsoft MPI)

    [*Win64-VS2015*](https://pan.baidu.com/s/1c2eX6lq) (OpenBLAS / Protobuf2.6 for VS2015 / CUDNN v7 / Microsoft MPI)

    [*Linux64*](https://pan.baidu.com/s/1c2ChKHy) (OpenMPI)

    For Windows, ``python27/35/36.lib`` should be copied to ``Dragon/3rdparty/lib``, it depends on the version of Python.

    For Linux, ``libpython-dev``, ``libprotobuf-dev``, ``libopenblas-dev`` and ``cuDNN`` should be installed by yourself.

4. Install Python Requirements

    ```Shell
    cd Dragon/python
    pip install -r requirements.txt
	```

5. Configure Dragon/CMakeLists.txt
	- Select optional libraries [PYTHON3 / CUDA / CUDNN / BLAS / SSE / MPI]
	- Set 3rdparty path (recommend to keep defualt)
	- Set Python include path & Numpy root path
	- Set CUDA compiling architectures if necessary
	- GCC version(4.8+, 5.0-) should add ``-std=c++11`` to ``CUDA_NVCC_FLAGS``, if ``nullptr`` is not found
	- We pre-generated files under ``Dragon/src/protos`` with protobuf-2.6, run ``protoc`` by yourself if higher are required

6. Environment Variables
    ### Linux(Only for OpenMPI):
	- Create dragon.conf

	    ```Shell
        sudo vim /etc/ld.so.conf.d/dragon.conf
        ```

		- Append 1 line for libraries dir of your 3rdparty, e.g. :
		 	- /home/Dragon/3rdparty/lib
	- rebuild the scaning cache

		```Shell
		sudo ldconfig
		```

	### Windows
	- add binary directionary to system environment variables, e.g. :
		- PATH=........;C:\Dragon\3rdparty\bin;


7. Setup MPI [Optional]
	#### Linux:
	- We use OpenMPI which supports "cuda-aware-mpi"
	- See more:
		- https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/
		- https://www.open-mpi.org/faq/?category=buildcuda
	- Run 3rdparty/setup_mpi.sh

		```Shell
		bash ./setup_mpi.sh
		```
    - Install

        ```Shell
		sudo cp 3rdparty/openmpi/install/bin/mpirun /usr/bin
		```
	#### Windows:
	- We use Microsoft MPI which can perfectly run at lastest Windows10
	- Microsoft MPI is intergrated into 3rdparty and you should do nothing

8. Compile
    #### Linux:
	- Install cmake

	    ```Shell
	    sudo apt-get install cmake
	    ```
	- Make

        ```Shell
        cd Dragon
        mkdir build
        cd build
        cmake ..
        make install -j16
        ```



	#### Windows:
	- Install cmake-gui
	- Mkdir Dragon/build
	- Configure and generate MSVC project in Dragon/build
	- Open Dragon/build/Dragon.sln
	- Compile and generate for "INSTALL" solution

8. Deploy

	```Shell
    cd Dragon/python
	python setup.py install
	```

	``Hint``: If you do not have permission, try as follows:

	```Shell
    cd Dragon/python
	python setup.py install --user
	```

----

##  Usage

### Import

```Shell
import dragon
```

### Virtual DL Frameworks

**------------------- Attention -------------------**

``tensorflow`` and ``theano`` are incomplete yet, prefer not to use it.

Currently, we recommend ``caffe`` and ``tiny-dragon``(ops + thenao.function + theano.tensor.grad + updaters)

**-------------------------------------------------**

```Shell
import dragon.vm.theano as theano
import dragon.vm.caffe as caffe
import dragon.vm.tensorflow as tf
```

### Tutorials

[IPython Notebook] -> (https://github.com/PhyscalX/Tutorials)

We will revise several classical examples, covering both CV, NLP and RL.

### Device

```Shell
import dragon.config
dragon.config.EnableCPU()
dragon.config.EnableCUDA(device_id, use_cudnn=True)
```

### Memonger

Dragon is a extremely memory efficient framework.

It is supported to drop intermediate results(mirrow stage) during forward phase, and share grads during backward phase,

takes 25% and 50% memory-usage comparing caffe and tensorflow respectively.

To use it, just:
 
```Shell
import dragon.memonger as opt
```

- ShareGrads

```Shell
opt.share_grads()
```

- Drop

```Shell
import dragon.ops as ops
y = opt.drop(ops.Relu, x)
```


### Scope

As a graph based framework, Dragon supports various scopes.

- NameScope

```Shell
import dragon
from dragon.core.tensor import Tensor
with dragon.name_scope(prefix='conv1'):
    w = Tensor('weight').Variable()    # named as conv1/weight
    b = Tensor('bias').Variable()      # named as conv1/bias
```

- DeviceScope

```Shell
import dragon
with dragon.deive_scope(deivce='gpu', id=0, use_cudnn=True):
    x = ops.Add(a, b)    # use /gpu:0 and cuDNN
```

- PhaseScope

```Shell
import dragon
import dragon.vm.theano as theano
 with dragon.phase_scope(phase='train'):
    f = theano.function(outputs=y)    # force the training phase even without gradients computation
```

## License and Citation

Dragon is released under the [BSD 2-Clause license](https://github.com/neopenx/Dragon/blob/master/LICENSE).

Please cite Dragon in your publications if it helps your research:

    @article{pan2017dragon,
      Author = {Pan, Ting},
      Journal = {arXiv preprint arXiv:1707.08265},
      Title = {Dragon: A Computation Graph Virtual Machine Based Deep Learning Framework},
      Year = {2017}
    }
