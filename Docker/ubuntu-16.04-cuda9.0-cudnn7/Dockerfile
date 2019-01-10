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
    libtbb-dev \
    libsdl2-dev \
    libnuma-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libopencv-dev \
    libboost-all-dev \
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

RUN git clone --recursive https://github.com/seetaresearch/Dragon.git && \
    mv Dragon/ThirdParty ./ && rm -rf Dragon

RUN cd ThirdParty/mpi && bash build.sh && rm -rf src *.gz && cp bin/mpirun /usr/bin

RUN git clone https://github.com/seetaresearch/Dragon.git && \
    cd Dragon/Dragon && mkdir build && cd build && cmake .. -DTHIRD_PARTY_DIR=/ThirdParty \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 -DWITH_MPI=ON -DWITH_NCCL=ON -DBUILD_CXX_API=ON && \
    make install -j $(nproc) && cd .. && rm -rf build && \
    cd python && python3 setup.py install

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip