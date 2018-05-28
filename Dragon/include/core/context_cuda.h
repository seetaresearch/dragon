// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_CONTEXT_CUDA_H_
#define DRAGON_CORE_CONTEXT_CUDA_H_

#include "core/common.h"
#include "core/context.h"
#include "utils/cuda_device.h"
#include "utils/cudnn_device.h"

namespace dragon {

#ifdef WITH_CUDA

/**************************************************************************
 *  cuXXX libraries wrapper "Context" as "Handle".
 *  It's well known that each "Context" binds to some "Devices" in OpenCL.
 *  So, we must create different handles to associate different devices or
    the computations will be dispatched to the same GPU.
 *  Read more: http://docs.nvidia.com/cuda/cublas/, Sec 2.1.2.
 *  Also, "Handle" is thread safe,
    it seems not necessary to create handles for different threads
 *************************************************************************/

class CUDAObject {
 public:
    CUDAObject(): cur_gpu(0) {
        for (int i = 0; i < MAX_GPUS; i++) {
            cublas_handle[i] = nullptr;
            curand_generator[i] = nullptr;
#ifdef WITH_CUDNN
            cudnn_handle[i] = nullptr;
#endif
        }
    }

    ~CUDAObject() {
        for (int i = 0; i < MAX_GPUS; i++) {
            if (cublas_handle[i]) cublasDestroy_v2(cublas_handle[i]);
            if (curand_generator[i]) curandDestroyGenerator(curand_generator[i]);
#ifdef WITH_CUDNN
            if (cudnn_handle[i]) cudnnDestroy(cudnn_handle[i]);
#endif
        }
    }

    int cur_gpu;
    cublasHandle_t cublas_handle[MAX_GPUS];
    curandGenerator_t curand_generator[MAX_GPUS];
#ifdef WITH_CUDNN
    cudnnHandle_t cudnn_handle[MAX_GPUS];
#endif
};

class CUDAContext {
 public:
    CUDAContext(const DeviceOption& option)
        : gpu_id_(option.device_id()),
          random_seed_(option.has_random_seed() ? option.random_seed() : 3) {
        CPUContext context(option);
        CHECK_EQ(option.device_type(), CUDA);
        cublas_handle();
        curand_generator();
#ifdef WITH_CUDNN
        cudnn_handle();
#endif
    }

    CUDAContext(const int gpu_id = 0)
        : gpu_id_(gpu_id), random_seed_(3) {
        CPUContext context;
        cublas_handle();
        curand_generator();
#ifdef WITH_CUDNN
        cudnn_handle();
#endif
    }

    void SwitchToDevice() {
        CUDA_CHECK(cudaSetDevice(gpu_id_));
        cuda_object_.cur_gpu = gpu_id_;
    }

    inline static void FinishDeviceCompution() {
        cudaStreamSynchronize(cudaStreamDefault);
        cudaError_t error = cudaGetLastError();
        CHECK_EQ(error, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(error);
    }

    inline static void* New(size_t nbytes) {
        void* data;
        cudaMalloc(&data, nbytes);
        CHECK(data) << "Malloc cuda mem: " << nbytes << " bytes failed.";
        return data;
    }

    inline static void Memset(size_t nbytes, void* ptr) { cudaMemset(ptr, 0, nbytes); }

    template<class DstContext, class SrcContext>
    inline static void Memcpy(size_t nbytes, void* dst, const void* src) {
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDefault));
    }

    template<class DstContext, class SrcContext>
    inline static void MemcpyAsync(size_t nbytes, void* dst, const void* src) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    inline static void Delete(void* data) { cudaFree(data); }

    template<typename T, class DstContext, class SrcContext>
    static void Copy(int n, T* dst, const T* src) {
        if (dst == src) return;
        Memcpy<SrcContext, DstContext>(n * sizeof(T), (void*)dst, (const void*)src);
    }

    cublasHandle_t& cublas_handle() {
        auto& handle = cuda_object_.cublas_handle[gpu_id_];
        if (handle)  {
            return handle;
        } else {
            DeviceGuard gurad(gpu_id_);
            CUBLAS_CHECK(cublasCreate_v2(&handle));
#if CUDA_VERSION >= 9000
            if (TENSOR_CORE_AVAILABLE()) cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
#endif
            return handle;
        }
    }

    curandGenerator_t& curand_generator() {
        auto& generator = cuda_object_.curand_generator[gpu_id_];
        if (generator) {
            return generator;
        } else {
            DeviceGuard gurad(gpu_id_);
            CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, random_seed_));
            return generator;
        }
    }

#ifdef WITH_CUDNN
    cudnnHandle_t cudnn_handle() {
        auto& handle = cuda_object_.cudnn_handle[gpu_id_];
        if (handle)  {
            return handle;
        } else{
            DeviceGuard gurad(gpu_id_);
            CUDNN_CHECK(cudnnCreate(&handle));
            return handle;
        }
    }
#endif

    static CUDAObject cuda_object_;

 private:
    int gpu_id_, random_seed_;
};

static inline cublasHandle_t& cublas_handle() {
    int cur_gpu = CUDAContext::cuda_object_.cur_gpu;
    CHECK(CUDAContext::cuda_object_.cublas_handle[cur_gpu] != nullptr);
    return CUDAContext::cuda_object_.cublas_handle[cur_gpu];
}

static inline curandGenerator_t& curand_generator() {
    int cur_gpu = CUDAContext::cuda_object_.cur_gpu;
    CHECK(CUDAContext::cuda_object_.curand_generator[cur_gpu] != nullptr);
    return CUDAContext::cuda_object_.curand_generator[cur_gpu];
}

#ifdef WITH_CUDNN
static inline cudnnHandle_t& cudnn_handle() {
    int cur_gpu = CUDAContext::cuda_object_.cur_gpu;
    CHECK(CUDAContext::cuda_object_.cudnn_handle[cur_gpu] != nullptr);
    return CUDAContext::cuda_object_.cudnn_handle[cur_gpu];
}

#endif

#else  // WITH_CUDA

class CUDAContext {
 public:
    CUDAContext(const DeviceOption& option) { CUDA_NOT_COMPILED; }
    CUDAContext(const int gpu_id = 0) { CUDA_NOT_COMPILED; }

    void SwitchToDevice() { CUDA_NOT_COMPILED; }
    void FinishDeviceCompution() { CUDA_NOT_COMPILED; }

    template<class DstContext, class SrcContext>
    static void Memcpy(size_t nbytes, void* dst, const void* src) { CUDA_NOT_COMPILED; }
};

#endif // WITH_CUDA

}    // namespace dragon

#endif    // DRAGON_CORE_CONTEXT_CUDA_H_