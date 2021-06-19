FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN \
  rm /etc/apt/sources.list.d/cuda.list && \
  apt-get update && apt-get install -y \
  --no-install-recommends \
  --allow-change-held-packages \
  build-essential \
  cmake \
  git \
  wget \
  unzip \
  ssh \
  vim \
  libudev-dev \
  libz-dev \
  libnuma-dev \
  libprotobuf-dev \
  protobuf-compiler \
  python3-pip \
  python3-dev \
  python3-pyqt4 \
  python3-tk \
  libnccl2 \
  libnccl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN \
  pip3 install --no-cache-dir --upgrade setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy \
  protobuf \
  kpl-dataset \
  opencv-python \
  Pillow

RUN \
  git clone --recursive https://github.com/seetaresearch/dragon.git && \
  mv dragon/third_party/* /opt && rm -rf dragon

RUN cd /opt/mpi && bash build.sh && rm -rf src *.gz && cp bin/mpirun /usr/bin

RUN \
  git clone https://github.com/seetaresearch/dragon.git && \
  cd dragon/dragon && mkdir build && cd build && \
  cmake .. \
  -DTHIRD_PARTY_DIR=/opt \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DUSE_MPI=ON \
  -DUSE_NCCL=ON \
  -DUSE_AVX2=ON \
  -DUSE_FMA=ON && \
  make install -j $(nproc) && \
  cd .. && rm -rf build && \
  python3 setup.py install

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
