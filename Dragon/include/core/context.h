// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_CONTEXT_H_
#define DRAGON_CORE_CONTEXT_H_

#include <random>
#include <ctime>

#include "common.h"
#include "utils/logging.h"

#ifdef WITH_CUDA
#include "utils/cuda_device.h"
#endif

namespace dragon {

class CPUObject{
public:
    unique_ptr<std::mt19937> rand_generator;
};

class CPUContext{
 public:
    CPUContext(): random_seed_(3) { generator(); }
    CPUContext(unsigned int random_seed): random_seed_(random_seed) { generator(); }
    CPUContext(const DeviceOption& option): random_seed_(option.has_random_seed() ? 
                                                         option.random_seed() : 3) { generator(); }
    virtual ~CPUContext() {}

    inline void SwitchToDevice() {}
    inline void FinishDeviceCompution() { return; }

    inline static void* New(size_t nbytes) {
        void* data;
#ifdef WITH_CUDA_HOST_MEN
        CUDA_CHECK(cudaMallocHost(&data, nbytes));
#else
        data = malloc(nbytes);
#endif
        CHECK(data) << "malloc mem: " << nbytes << " bytes failed.";
        return data;
    }

    inline static void Memset(size_t nbytes, void* ptr) { memset(ptr, 0, nbytes); }
    template<class DstContext, class SrcContext>
    inline static void Memcpy(size_t nbytes, void* dst, const void* src) { memcpy(dst, src, nbytes); }
    inline static void Delete(void* data) { free(data); }

    template<typename T, class DstContext, class SrcContext>
    inline static void Copy(int n, T* dst, const T* src){
        if (dst == src) return;
        //  only the basic types(e.g. int/float) can memcpy correctly
        if (std::is_fundamental<T>::value)
            Memcpy<DstContext, SrcContext>(n * sizeof(T), (void*)dst, (const void*)src);
        else for (int i = 0; i < n; i++) dst[i] = src[i];
    }

    inline std::mt19937* generator() {
        auto& generator = cpu_object_.rand_generator;
        if (!generator.get()) 
            generator.reset(new std::mt19937(random_seed_));
        return generator.get();
    }

    static CPUObject cpu_object_;

 private:
    unsigned int random_seed_;
};

static inline std::mt19937* rand_generator() {
    return CPUContext::cpu_object_.rand_generator.get();
}

}    // namepsace dragon

#endif    // DRAGON_CORE_CONTEXT_H_