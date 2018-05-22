FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    python3-pip \
    python3-dev \
    python3-pyqt4 \
    python3-tk \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade setuptools wheel && \
    pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy \
    protobuf \ 
    lmdb \ 
    opencv-python \
    six \ 
    Pillow
    matplotlib \
    pyyaml

RUN wget http://dragon.seetatech.com/download/docker/ubuntu-16.04-cpu-openblas/3rdparty.zip && \
    unzip 3rdparty.zip && rm 3rdparty.zip && cd 3rdparty && bash ./setup_mpi.sh && rm -f *.gz && \
    cd openmpi && ls | grep -v install | xargs rm -r && cp install/bin/mpirun /usr/bin

RUN git clone https://github.com/seetaresearch/Dragon.git && \
    cd Dragon/Dragon && rm CMakeLists.txt && \
    wget http://dragon.seetatech.com/download/docker/ubuntu-16.04-cpu-openblas/CMakeLists.txt && \
    mkdir build && cd build && cmake .. && make install -j8 && cd .. && rm -rf build && \
    cd python && python3 setup.py install