FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

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
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir numpy protobuf

RUN git clone --recursive https://github.com/seetaresearch/dragon.git

RUN cd dragon && mkdir build && cd build && \
  cmake .. -DUSE_CUDA=OFF -DUSE_CUDNN=OFF

RUN cd dragon/build && make install -j $(nproc)
RUN pip install ./dragon && rm -rf dragon/build
