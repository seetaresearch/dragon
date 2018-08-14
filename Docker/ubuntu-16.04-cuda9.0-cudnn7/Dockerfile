FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    ssh \
    vim \
    libnuma-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libopenblas-dev \
    libnccl2 \
    libnccl-dev \
    python3-pip \
    python3-dev \
    python3-pyqt4 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy \
    protobuf \ 
    lmdb \ 
    opencv-python \
    six \ 
    Pillow \
    matplotlib \
    scikit-image \
    pyyaml \
    cython

RUN wget http://dragon.seetatech.com/download/docker/ubuntu-16.04-cuda9.0-cudnn7/3rdparty.zip && \
    unzip 3rdparty.zip && rm 3rdparty.zip && cd 3rdparty && bash ./setup_mpi.sh && rm -f *.gz && \
    cd openmpi && ls | grep -v install | xargs rm -r && cp install/bin/mpirun /usr/bin

RUN git clone https://github.com/seetaresearch/Dragon.git && \
    cd Dragon/Dragon && rm CMakeLists.txt && \
    wget http://dragon.seetatech.com/download/docker/ubuntu-16.04-cuda9.0-cudnn7/CMakeLists.txt && \
    mkdir build && cd build && cmake .. && make install -j8 && cd .. && rm -rf build && \
    cd python && python3 setup.py install