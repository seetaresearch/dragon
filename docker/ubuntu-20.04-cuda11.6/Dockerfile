FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  wget \
  unzip \
  ssh \
  vim \
  libz-dev \
  libprotobuf-dev \
  protobuf-compiler \
  python3-pip \
  python3-dev \
  libnccl2 \
  libnccl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir numpy protobuf

RUN git clone --recursive https://github.com/seetaresearch/dragon.git

RUN cd dragon/third_party/mpi && bash build.sh && rm -rf src *.gz && cp bin/mpirun /usr/bin
RUN ln -s /usr/include/cudnn* /usr/local/cuda/include
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64

RUN cd dragon && mkdir build && cd build && \
  cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_MPI=ON -DUSE_NCCL=ON 

RUN cd dragon/build && make install -j $(nproc)
RUN pip install ./dragon && rm -rf dragon/build
