# Dragon: A Computation Graph Virtual Machine Based Deep Learning Framework

### Compile Requirements for C++

0. Google Protocol Buffer
1. Python (2.7, 64bit) &nbsp; | &nbsp; Anaconda (2.7, 64bit)
2. CUDA [Optional]
3. CUDNN [Optional]
4. OpenMPI [Optional]

-----
### Runtime Requirements for Python

0. Package: protobuf
1. Package: lmdb

-----
### Installation
1. Clone this repository

2. (Optional) Download and install [CUDA](https://developer.nvidia.com/cuda-toolkit)

      (Optional) Download and install [CUDNN](https://developer.nvidia.com/cudnn)

3. (Optional) Download 3rdparty.zip and unzip to Dragon/3rdparty (Out of source code dir)

    [*Win64*](https://pan.baidu.com/s/1pLmGOLt) (OpenBLAS / Protobuf for VS2013 / CUDNN v6 / Microsoft MPI)

    [*Linux64*](https://pan.baidu.com/s/1qXPEOWG) (OpenMPI)

4. Configure Dragon/CMakeLists.txt
	- Select optional libraries [CUDA / CUDNN / BLAS / SSE / MPI / MPI_CUDA_AWARE / CUDA_FP16]
	- Set 3rdparty path (recommend to keep defualt)
	- Set python & numpy root path
	- Set cuda compiling architectures if necessary
	- GCC version(4.8+, 5.0-) should add ``-std=c++11`` to ``CUDA_NVCC_FLAGS``, if ``nullptr`` is not found.

5. Environment Variables
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


6. Setup MPI [Optional]
	#### Linux:
	- We use OpenMPI which support "cuda-aware-mpi"
	- See more:
		- https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/
		- https://www.open-mpi.org/faq/?category=buildcuda
	- Run 3rdparty/setup_mpi.sh

		```Shell
		sudo ./setup_mpi.sh
		```

	#### Windows:
	- We use Microsoft MPI which can perfectly run at lastest Windows10
	- Microsoft MPI is intergrated into 3rdparty and you should do nothing

7. Compile
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
	python Dragon/setup.py install
	```

----

##  Usage

### Import

```Shell
import dragon
```

### Virtual DL Frameworks

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

### Automatic Memory Optimization(AMC)

```Shell
import dragon.config
dragon.config.SetDebugMode(False)
```

This option will make all gradients share a global tensor(debugging is intractable).

which prefers a 50% memory-usage and 15% slower solution during training phase.

### Scope

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
