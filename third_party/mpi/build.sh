#!/bin/sh

# See https://www.open-mpi.org/faq/?category=buildcuda
USE_CUDA_AWARE=${1:-1}

if [ -f "openmpi-4.0.0.tar.gz" ];then echo "Skip downloading...."
else wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
fi

rm -rf src
tar -xzvf openmpi-4.0.0.tar.gz
mv openmpi-4.0.0 src
cd src

INSTALL_PATH=$(cd "$(dirname "$0")/..";pwd)

if [ $USE_CUDA_AWARE -eq 1 ];then 
echo "Build with cuda...."
read -p "Press any key to continue." var
./configure CFLAGS=-fPIC \
            CXXFLAGS=-fPIC \
            --with-cuda \
            --with-pic=PIC \
            --without-verbs \
            --without-ucx \
            --enable-shared \
            --enable-static \
            --enable-mpi-thread-multiple \
            --prefix=$INSTALL_PATH
else 
echo "Build without cuda...."
read -p "Press any key to continue." var
./configure CFLAGS=-fPIC \
            CXXFLAGS=-fPIC \
            --with-pic=PIC \
            --without-verbs \
            --without-ucx \
            --enable-shared \
            --enable-static \
            --enable-mpi-thread-multiple \
            --prefix=$INSTALL_PATH
fi
make install -j $(getconf _NPROCESSORS_ONLN)
