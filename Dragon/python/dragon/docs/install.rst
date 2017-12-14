Installing Dragon
=================

Get the Latest Version
----------------------

**$** Clone the `Dragon`_ repository

.. code-block:: shell

   git clone https://github.com/neopenx/Dragon.git

We will call the directory that you cloned Dragon as ``REPO_ROOT``.

Besides, let's call the ``REPO_ROOT/Dragon`` as ``DRAGON_ROOT``.

Installation - Linux (Normal, CPU)
----------------------------------

**Step 1:** Install C++ Dependencies

**$** Setup Python Development Environment

.. code-block:: shell

    sudo apt-get install libpython-dev

**Note:** You can also use `Anaconda`_, A powerful toolkit for Data Science.

**$** Setup C++ Development Environment

    sudo apt-get install libprotobuf-dev
    sudo apt-get install protobuf-compiler
    sudo apt-get install libopenblas-dev

**Step 2:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 3:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``BLAS`` / ``SSE``]

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**Step 4:** Compile Dragon

**$** Install CMake

.. code-block:: shell

    sudo apt-get install cmake

**$** Make

.. code-block:: shell

    cd $DRAGON_ROOT
    mkdir build
    cd build
    cmake ..
    make install -j16

**Step 5:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user



Installation - Linux (Normal, GPU)
----------------------------------

**Step 1:** Preparing GPU Prerequisites

**$** Download and install `CUDA`_

**$** (Optional) Download and install `CUDNN`_

**Step 2:** Install C++ Dependencies

**$** Setup Python Development Environment

.. code-block:: shell

    sudo apt-get install libpython-dev

**Note:** You can also use `Anaconda`_, A powerful toolkit for Data Science.

**$** Setup C++ Development Environment

    sudo apt-get install libprotobuf-dev
    sudo apt-get install protobuf-compiler
    sudo apt-get install libopenblas-dev

**Step 3:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 4:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``CUDA`` / ``CUDNN`` / ``BLAS`` / ``SSE``]

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**$** Set CUDA compiling architectures if necessary

**$** GCC version(4.8+, 5.0-) should add ``-std=c++11`` to ``CUDA_NVCC_FLAGS``, if ``nullptr`` is not found

**Step 5:** Compile Dragon

**$** Install CMake

.. code-block:: shell

    sudo apt-get install cmake

**$** Make

.. code-block:: shell

    cd $DRAGON_ROOT
    mkdir build
    cd build
    cmake ..
    make install -j16

**Step 6:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user



Installation - Linux (Distributed, CPU)
---------------------------------------

**Step 1:** Download ``3rdparty.zip`` and unzip it under ``REPO_ROOT``

`3rdparty_linux_x64.zip <http://dragon.seetatech.com/download/3rdparty_linux_x64_dist_cpu.zip>`_ (OpenMPI)

**Step 2:** Install C++ Dependencies

**$** Setup Python Development Environment

.. code-block:: shell

    sudo apt-get install libpython-dev

**Note:** You can also use `Anaconda`_, A powerful toolkit for Data Science.

**$** Setup C++ Development Environment

    sudo apt-get install libprotobuf-dev
    sudo apt-get install protobuf-compiler
    sudo apt-get install libopenblas-dev

**Step 3:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 4:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``BLAS`` / ``SSE`` / ``MPI``]

**$** Set ``3RDPARTY_DIR`` (Recommend to Keep Default)

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**Step 5:** Setup MPI

.. code-block:: shell

    cd $REPO_ROOT/3rdparty
    bash ./setup_mpi.sh
    sudo cp openmpi/install/bin/mpirun /usr/bin

**Step 6:** Compile Dragon

**$** Install CMake

.. code-block:: shell

    sudo apt-get install cmake

**$** Make

.. code-block:: shell

    cd $DRAGON_ROOT
    mkdir build
    cd build
    cmake ..
    make install -j16

**Step 7:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user



Installation - Linux (Distributed, GPU)
---------------------------------------

**Step 1:** Preparing GPU Prerequisites

**$** Download and install `CUDA`_

**$** (Optional) Download and install `CUDNN`_

**$** (Optional) Download and install `NCCL`_

**Step 2:**  Download ``3rdparty.zip`` and unzip it under ``REPO_ROOT``

`3rdparty_linux_x64.zip <http://dragon.seetatech.com/download/3rdparty_linux_x64_dist_gpu.zip>`_ (OpenMPI)

**Step 3:** Install C++ Dependencies

**$** Setup Python Development Environment

.. code-block:: shell

    sudo apt-get install libpython-dev

**Note:** You can also use `Anaconda`_, A powerful toolkit for Data Science.

**$** Setup C++ Development Environment

    sudo apt-get install libprotobuf-dev
    sudo apt-get install protobuf-compiler
    sudo apt-get install libopenblas-dev

**Step 4:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 5:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``CUDA`` / ``CUDNN`` / ``BLAS`` / ``SSE`` / ``MPI``]

**$** Set ``3RDPARTY_DIR`` (Recommend to Keep Default)

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**$** Set CUDA compiling architectures if necessary

**$** GCC version(4.8+, 5.0-) should add ``-std=c++11`` to ``CUDA_NVCC_FLAGS``, if ``nullptr`` is not found

**$** OpenMPI can take ``NCCL`` and our ``CUDA-AWARE`` communications at the same time.

**Step 6:** Setup MPI

.. code-block:: shell

    cd $REPO_ROOT/3rdparty
    bash ./setup_mpi.sh
    sudo cp openmpi/install/bin/mpirun /usr/bin

**Step 7:** Compile Dragon

**$** Install CMake

.. code-block:: shell

    sudo apt-get install cmake

**$** Make

.. code-block:: shell

    cd $DRAGON_ROOT
    mkdir build
    cd build
    cmake ..
    make install -j16

**Step 8:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user


Installation - Windows (Normal, CPU)
------------------------------------

**Step 1:**  Download ``3rdparty.zip`` and unzip it under ``REPO_ROOT``

`3rdparty_vc12_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc12_x64_cpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2013)

`3rdparty_vc14_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc14_x64_cpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2015)

**$** You must copy ``python27/35/36.lib`` to ``REPO_ROOT/3rdparty/lib``, it depends on the version of Python

**Step 2:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 3:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``BLAS`` / ``SSE``]

**$** Set ``3RDPARTY_DIR`` (Recommend to Keep Default)

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**Step 4:** Set Environment Variables

Add ``REPO_ROOT/3rdparty/bin`` to system environment variables

.. code-block:: shell

    PATH=........;C:\xyz\Dragon\3rdparty\bin;

**Step 5:** Compile Dragon

**$** Install `CMake-GUI <https://cmake.org>`_

**$** Make ``build`` directory under ``DRAGON_ROOT``

**$** Configure and generate MSVC project in ``DRAGON_ROOT/build``

**$** Open ``DRAGON_ROOT/build/Dragon.sln``

**$** Compile and generate for ``INSTALL`` solution

**Step 6:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user



Installation - Windows (Normal, GPU)
------------------------------------

**Step 1:** Preparing GPU Prerequisites

**$** Download and install `CUDA`_

**$** (Optional) Download and install `CUDNN`_

**Step 2:** Download ``3rdparty.zip`` and unzip it under ``REPO_ROOT``

`3rdparty_vc12_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc12_x64_gpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2013)

`3rdparty_vc14_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc14_x64_gpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2015)

**$** You must copy ``python27/35/36.lib`` to ``REPO_ROOT/3rdparty/lib``, it depends on the version of Python

**$** Recommend you to install ``cuDNN`` into ``REPO_ROOT/3rdparty``

**Step 3:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 4:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``CUDA`` / ``CUDNN`` / ``BLAS`` / ``SSE``]

**$** Set ``3RDPARTY_DIR`` (Recommend to Keep Default)

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**$** Set CUDA compiling architectures if necessary

**Step 5:** Set Environment Variables

Add ``REPO_ROOT/3rdparty/bin`` to system environment variables

.. code-block:: shell

    PATH=........;C:\xyz\Dragon\3rdparty\bin;

**Step 6:** Compile Dragon

**$** Install `CMake-GUI <https://cmake.org>`_

**$** Make ``build`` directory under ``DRAGON_ROOT``

**$** Configure and generate MSVC project in ``DRAGON_ROOT/build``

**$** Open ``DRAGON_ROOT/build/Dragon.sln``

**$** Compile and generate for ``INSTALL`` solution

**Step 7:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user



Installation - Windows (Distributed, CPU)
-----------------------------------------

**Step 1:**  Download ``3rdparty.zip`` and unzip it under ``REPO_ROOT``

`3rdparty_vc12_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc12_x64_dist_cpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2013 / Microsoft MPI)

`3rdparty_vc14_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc14_x64_dist_cpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2015 / Microsoft MPI)

**$** You must copy ``python27/35/36.lib`` to ``REPO_ROOT/3rdparty/lib``, it depends on the version of Python

**Step 2:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 3:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``BLAS`` / ``SSE`` / ``MPI``]

**$** Set ``3RDPARTY_DIR`` (Recommend to Keep Default)

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**Step 4:** Set Environment Variables

Add ``DRAGON_ROOT/3rdparty/bin`` to system environment variables

.. code-block:: shell

    PATH=........;C:\xyz\Dragon\3rdparty\bin;

**Step 5:** Compile Dragon

**$** Install `CMake-GUI <https://cmake.org>`_

**$** Make ``build`` directory under ``DRAGON_ROOT``

**$** Configure and generate MSVC project in ``DRAGON_ROOT/build``

**$** Open ``DRAGON_ROOT/build/Dragon.sln``

**$** Compile and generate for ``INSTALL`` solution

**Step 6:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user



Installation - Windows (Distributed, GPU)
-----------------------------------------

**Step 1:** Preparing GPU Prerequisites

**$** Download and install `CUDA`_

**$** (Optional) Download and install `CUDNN`_

**Step 2:** Download ``3rdparty.zip`` and unzip it under ``REPO_ROOT``

`3rdparty_vc12_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc12_x64_dist_gpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2013 / Microsoft MPI)

`3rdparty_vc14_x64.zip <http://dragon.seetatech.com/download/3rdparty_vc14_x64_dist_gpu.zip>`_ (OpenBLAS / Google Protobuf 2.6 For VS2015 / Microsoft MPI)

**$** You must copy ``python27/35/36.lib`` to ``REPO_ROOT/3rdparty/lib``, it depends on the version of Python

**$** Recommend you to install ``cuDNN`` into ``REPO_ROOT/3rdparty``

**Step 3:** Install Python Requirements

.. code-block:: shell

    cd $DRAGON_ROOT/python
    pip install -r requirements.txt

**Step 4:** Configure ``DRAGON_ROOT/CMakeLists.txt``

**$** Select optional libraries [``PYTHON3`` / ``CUDA`` / ``CUDNN`` / ``BLAS`` / ``SSE`` / ``MPI``]

**$** Set ``3RDPARTY_DIR`` (Recommend to Keep Default)

**$** Set ``PYTHON_INCLUDE_DIR`` / ``ANACONDA_ROOT_DIR`` and ``NUMPY_ROOT_DIR``

**$** Set CUDA compiling architectures if necessary

**Step 5:** Set Environment Variables

Add ``REPO_ROOT/3rdparty/bin`` to system environment variables

.. code-block:: shell

    PATH=........;C:\xyz\Dragon\3rdparty\bin;

**Step 6:** Compile Dragon

**$** Install `CMake-GUI <https://cmake.org>`_

**$** Make ``build`` directory under ``DRAGON_ROOT``

**$** Configure and generate MSVC project in ``DRAGON_ROOT/build``

**$** Open ``DRAGON_ROOT/build/Dragon.sln``

**$** Compile and generate for ``INSTALL`` solution

**Step 7:** Install Dragon

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install

**Note:** If you do not have permission, try as follows:

.. code-block:: shell

    cd $DRAGON_ROOT/python
    python setup.py install --user


.. _Anaconda: https://www.anaconda.com/download
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _CUDNN: https://developer.nvidia.com/cudnn
.. _NCCL: https://developer.nvidia.com/nccl
.. _Dragon: https://github.com/neopenx/Dragon
